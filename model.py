from typing import Dict, List, Any, Tuple, Union

import torch
import numpy
from overrides import overrides
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.attention import AdditiveAttention
from allennlp.modules import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import Metric, BLEU


@Model.register("ms_pointer_network")
class MSPointerNetwork(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder_1: TextFieldEmbedder,
                 source_encoder_1: Seq2SeqEncoder,
                 beam_size: int,
                 max_decoding_steps: int,
                 decoder_output_dim: int,
                 target_embedding_dim: int = 30,
                 namespace: str = "tokens",
                 tensor_based_metric: Metric = None,
                 align_embeddings: bool = True,
                 source_embedder_2: TextFieldEmbedder = None,
                 source_encoder_2: Seq2SeqEncoder = None) -> None:
        super().__init__(vocab)
        self._source_embedder_1 = source_embedder_1
        self._source_embedder_2 = source_embedder_1 or self._source_embedder_1
        self._source_encoder_1 = source_encoder_1
        self._source_encoder_2 = source_encoder_2 or self._source_encoder_1
        
        self._source_namespace = namespace
        self._target_namespace = namespace
        
        self.encoder_output_dim_1 = self._source_encoder_1.get_output_dim()
        self.encoder_output_dim_2 = self._source_encoder_2.get_output_dim()
        self.cated_encoder_out_dim = self.encoder_output_dim_1 + self.encoder_output_dim_2
        self.decoder_output_dim = decoder_output_dim

        # TODO: AllenNLP实现的Addictive Attention可能没有bias
        self._attention_1 = AdditiveAttention(self.decoder_output_dim, self.encoder_output_dim_1)
        self._attention_2 = AdditiveAttention(self.decoder_output_dim, self.encoder_output_dim_2)

        if not align_embeddings:
            self.target_embedding_dim = target_embedding_dim
            self._target_vocab_size = self.vocab.get_vocab_size(namespace=self._target_namespace)
            self._target_embedder = Embedding(self._target_vocab_size, target_embedding_dim)
        else:
            self._target_embedder = self._source_embedder_1._token_embedders["tokens"]
            self._target_vocab_size = self.vocab.get_vocab_size(namespace=self._target_namespace)
            self.target_embedding_dim = self._target_embedder.get_output_dim()

        self.decoder_input_dim = self.encoder_output_dim_1 + self.encoder_output_dim_2 + \
                                 self.target_embedding_dim

        self._decoder_cell = LSTMCell(self.decoder_input_dim, self.decoder_output_dim)

        # 用于将两个encoder的最后隐层状态映射成解码器初始状态
        self._encoder_out_projection_layer = torch.nn.Linear(
                in_features=self.cated_encoder_out_dim,
                out_features=self.decoder_output_dim)  #  TODO: bias - true of false?

        # 软门控机制参数，用于计算lambda
        self._gate_projection_layer = torch.nn.Linear(
                in_features=self.decoder_output_dim + self.decoder_input_dim,
                out_features=1, bias=False)

        self._start_index = self.vocab.get_token_index(START_SYMBOL, namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, namespace)
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token, namespace)
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        self._tensor_based_metric = tensor_based_metric or \
            BLEU(exclude_indices={self._pad_index, self._end_index, self._start_index})
        
    def _encode(self,
                source_tokens_1: Dict[str, torch.Tensor],
                source_tokens_2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        分别将source1和source2的token ids经过encoder编码，输出各自的mask以及encoder_out。
        同时token_ids信息也会附加。
        """
        
        # 1. 编码source1
        # shape: (batch_size, seq_max_len_1)
        source_mask_1 = util.get_text_field_mask(source_tokens_1)
        # shape: (batch_size, seq_max_len_1, encoder_input_dim_1)
        embedder_out_1 = self._source_embedder_1(source_tokens_1)
        # shape: (batch_size, seq_max_len_1, encoder_output_dim_1)
        encoder_out_1 = self._source_encoder_1(embedder_out_1, source_mask_1)

        # 2. 编码source2
        # shape: (batch_size, seq_max_len_2)
        source_mask_2 = util.get_text_field_mask(source_tokens_2)
        # shape: (batch_size, seq_max_len_2, encoder_input_dim_2)
        embedder_out_2 = self._source_embedder_2(source_tokens_2)
        # shape: (batch_size, seq_max_len_2, encoder_input_dim_2)
        encoder_out_2 = self._source_encoder_2(embedder_out_2, source_mask_2)

        return {
            "source_mask_1": source_mask_1,
            "source_mask_2": source_mask_2,
            "source_token_ids_1": source_tokens_1["tokens"],
            "source_token_ids_2": source_tokens_2["tokens"],
            "encoder_out_1": encoder_out_1,
            "encoder_out_2": encoder_out_2,
        }
    
    def _init_decoder_state(self,
                            state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        初始化decoder：更新传入的state，使之带有decoder的context和hidden向量。
                      其中hidden向量（h_0）通过两个编码器的最终隐层状态经过一个
                      映射得到，context初始化为0向量。
        """
        batch_size = state["encoder_out_1"].size()[0]

        # 根据每个batch的mask情况，获取最终rnn隐层状态
        # shape: (batch_size, encoder_output_dim_1)
        encoder_final_output_1 = util.get_final_encoder_states(
                state["encoder_out_1"],
                state["source_mask_1"],
                self._source_encoder_1.is_bidirectional())
        # shape: (batch_size, encoder_output_dim_2)
        encoder_final_output_2 = util.get_final_encoder_states(
                state["encoder_out_2"],
                state["source_mask_2"],
                self._source_encoder_2.is_bidirectional())

        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = torch.relu(self._encoder_out_projection_layer(
                torch.cat([encoder_final_output_1, encoder_final_output_2], dim=-1)))
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["decoder_hidden"].new_zeros(batch_size, self.decoder_output_dim)
        
        return state

    @overrides
    def forward(self,
                source_tokens_1: Dict[str, torch.LongTensor],
                source_tokens_2: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        
        # 分成训练、验证/测试、预测，这三种情况分别考虑

        # 1. 训练时：必然同时提供了target_tokens作为ground truth。
        #    此时，只需要计算loss，无需beam search
        if self.training:
            assert target_tokens is not None
            
            state = self._encode(source_tokens_1, source_tokens_2)
            state["target_token_ids"] = target_tokens["tokens"]
            state = self._init_decoder_state(state)
            output_dict = self._forward_loss(target_tokens, state)
            output_dict["metadata"] = metadata
            return output_dict  # 包含loss、metadata两项

        # 2. 验证/测试时：self.training为false，但是提供了target_tokens。
        #    此时，需要计算loss、运行beam search、计算评价指标
        elif target_tokens:
            
            # 计算loss
            state = self._encode(source_tokens_1, source_tokens_2)
            state["target_token_ids"] = target_tokens["tokens"]
            state = self._init_decoder_state(state)
            output_dict = self._forward_loss(target_tokens, state)
            
            # 运行beam search
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

            # 计算评价指标（BLEU）
            if self._tensor_based_metric is not None:
                # shape: (batch_size, beam_size, max_decoding_steps)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_decoding_steps)
                best_predictions = top_k_predictions[:, 0, :]
                # shape: (batch_size, target_seq_len)
                gold_tokens = target_tokens["tokens"]
                self._tensor_based_metric(best_predictions, gold_tokens)
            output_dict["metadata"] = metadata
            return output_dict  # 包含loss、metadata、top-k、top-k log prob四项

        # 3. 预测时：self.training为false，同时也没有提供target_tokens。
        #    此时，只需要运行beam search执行top-k预测即可
        else:
            state = self._encode(source_tokens_1, source_tokens_2)
            state = self._init_decoder_state(state)
            output_dict = {"metadata": metadata}
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            return output_dict  # 包含metadata、top-k、top-k log prob三项

    def _forward_loss(self,
                      target_tokens: Dict[str, torch.Tensor],
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        为输入的一个batch计算损失（仅在训练时调用）。
        """
        batch_size, target_seq_len = target_tokens["tokens"].size()

        # shape: (batch_size, seq_max_len_1)
        source_mask_1 = state["source_mask_1"]
        # shape: (batch_size, seq_max_len_2)
        source_mask_2 = state["source_mask_2"]

        # 需要生成的最大步数永远比目标序列（<start> ... <end>）的最大长度少1步
        num_decoding_steps = target_seq_len - 1

        step_log_likelihoods = []  # 存放每个时间步，目标词的log似然值
        for timestep in range(num_decoding_steps):  # t: 0..T

            # 当前时刻要输入的token id，shape (batch_size,)
            input_choices = target_tokens["tokens"][:, timestep]

            # 更新一步解码器状态（计算各类中间变量，例如attention分数、软门控分数）
            state = self._decoder_step(input_choices, state)

            # 获取decoder_hidden相对于两个source的attention分数
            # shape: (batch_size, seq_max_len_1)
            attentive_weights_1 = state["attentive_weights_1"]
            # shape: (batch_size, seq_max_len_2)
            attentive_weights_2 = state["attentive_weights_2"]

            # 计算target_to_source，指明当前要输出的target (ground truth)，是否出现在source之中
            # shape: (batch_size, seq_max_len_1)
            target_to_source_1 = (state["source_token_ids_1"] == 
                    state["target_token_ids"][:, timestep+1].unsqueeze(-1))
            # shape: (batch_size, seq_max_len_2)
            target_to_source_2 = (state["source_token_ids_2"] ==
                    state["target_token_ids"][:, timestep+1].unsqueeze(-1))

            # 根据上面的信息计算当前时间步target token的对数似然
            step_log_likelihood = self._get_ll_contrib(attentive_weights_1,
                    attentive_weights_2,
                    source_mask_1,
                    source_mask_2,
                    target_to_source_1,
                    target_to_source_2,
                    state["target_token_ids"][:, timestep + 1],
                    state["gate_score"])
            step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))
        
        # 将各个时间步的对数似然合并成一个tensor
        # shape: (batch_size, num_decoding_steps = target_seq_len - 1)
        log_likelihoods = torch.cat(step_log_likelihoods, 1)

        # 获取包含START和END的target mask
        # shape: (batch_size, target_seq_len)
        target_mask = util.get_text_field_mask(target_tokens)

        # 去掉第一个，不会作为目标词的START
        # shape: (batch_size, num_decoding_steps = target_seq_len - 1)
        target_mask = target_mask[:, 1:].float()

        # 将各个时间步上的对数似然tensor使用mask累加，得到整个时间序列的对数似然
        log_likelihood = (log_likelihoods * target_mask).sum(dim=-1)

        loss = - log_likelihood.sum() / batch_size

        return {"loss": loss}

    def _decoder_step(self,
                      last_predictions: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        更新一步decoder状态。
        """

        # shape: (group_size, seq_max_len_1, encoder_output_dim_1)
        source_mask_1 = state["source_mask_1"].float()
        # shape: (group_size, seq_max_len_2, encoder_output_dim_2)
        source_mask_2 = state["source_mask_2"].float()
        # y_{t-1}, shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        # a_t, shape: (group_size, seq_max_len_1)
        state["attentive_weights_1"] = self._attention_1(
                state["decoder_hidden"], state["encoder_out_1"], source_mask_1)
        # a'_t, shape: (group_size, seq_max_len_2)
        state["attentive_weights_2"] = self._attention_2(
                state["decoder_hidden"], state["encoder_out_2"], source_mask_2)

        # c_t, shape: (group_size, encoder_output_dim_1)
        attentive_read_1 = util.weighted_sum(state["encoder_out_1"], state["attentive_weights_1"])
        # c'_t, shape: (group_size, encoder_output_dim_2)
        attentive_read_2 = util.weighted_sum(state["encoder_out_2"], state["attentive_weights_2"])

        # 计算软门控机制：lambda
        # shape: (group_size, target_embedding_dim + encoder_output_dim_1 + encoder_output_dim_2 + decoder_output_dim)
        gate_input = torch.cat((embedded_input, attentive_read_1, attentive_read_2,
                state["decoder_hidden"]), dim=-1)
        # shape: (group_size,)
        gate_projected = self._gate_projection_layer(gate_input).squeeze(-1)
        # shape: (group_size,)
        state["gate_score"] = torch.sigmoid(gate_projected)

        # shape: (group_size, target_embedding_dim + encoder_output_dim_1 + encoder_output_dim_2)
        decoder_input = torch.cat((embedded_input, attentive_read_1, attentive_read_2), dim=-1)

        # 更新decoder状态（hidden和context/cell）
        state["decoder_hidden"], state["decoder_context"] = self._decoder_cell(
                decoder_input, (state["decoder_hidden"], state["decoder_context"]))
        
        return state

    def _get_ll_contrib(self,
                        copy_scores_1: torch.Tensor,
                        copy_scores_2: torch.Tensor,
                        source_mask_1: torch.Tensor,
                        source_mask_2: torch.Tensor,
                        target_to_source_1: torch.Tensor,
                        target_to_source_2: torch.Tensor,
                        target_tokens: torch.Tensor,
                        gate_score: torch.Tensor) -> torch.Tensor:
        """
        根据一个时间步的attention分数、黄金token，计算黄金token的对数似然。

        参数：
            - copy_scores_1：对第一个source的注意力分值。
                    shape: (batch_size, seq_max_len_1)
            - copy_scores_2：对第二个source的注意力分值。
                    shape: (batch_size, seq_max_len_2)
            - source_mask_1：第一个source的mask
                    shape: (batch_size, seq_max_len_1)
            - source_mask_2：第二个source的mask
                    shape: (batch_size, seq_max_len_2)
            - target_to_source_1：目标词是否为第一个source对应位置的词
                    shape: (batch_size, seq_max_len_1)
            - target_to_source_2：目标词是否为第二个source对应位置的词
                    shape: (batch_size, seq_max_len_2)
            - target_tokens：当前时间步的目标词
                    shape: (batch_size,)
            - gate_score：从第一个source拷贝词语的概率（0-1之间）
                    shape: (batch_size,)

        返回：
            当前时间步，生成目标词的对数似然（log-likelihood）
                    shape: (batch_size,)
        """
        # 计算第一个source的分值
        # shape: (batch_size, seq_max_len_1)
        combined_log_probs_1 = (copy_scores_1 + 1e-45).log() + (target_to_source_1.float()
                + 1e-45).log() + (source_mask_1.float() + 1e-45).log()
        # shape: (batch_size,)
        log_probs_1 = util.logsumexp(combined_log_probs_1)  # log(exp(a[0]) + ... + exp(a[L]))

        # 计算第二个source的分值
        # shape: (batch_size, seq_max_len_2)
        combined_log_probs_2 = (copy_scores_2 + 1e-45).log() + (target_to_source_2.float()
                + 1e-45).log() + (source_mask_2.float() + 1e-45).log()
        # shape: (batch_size,)
        log_probs_2 = util.logsumexp(combined_log_probs_2)  # log(exp(a[0]) + ... + exp(a[L]))

        # 计算 log(p1 * gate + p2 * (1-gate))
        log_gate_score_1 = gate_score.log()  # shape: (batch_size,)
        log_gate_score_2 = (1 - gate_score).log()  # shape: (batch_size,)
        item_1 = (log_gate_score_1 + log_probs_1).unsqueeze(-1)  # shape: (batch_size, 1)
        item_2 = (log_gate_score_2 + log_probs_2).unsqueeze(-1)  # shape: (batch_size, 1)
        step_log_likelihood = util.logsumexp(torch.cat((item_1, item_2), -1))  # shape: (batch_size,)
        return step_log_likelihood

    def _forward_beam_search(self,
                             state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask_1"].size()[0]
        start_predictions = state["source_mask_1"].new_full((batch_size,), fill_value=self._start_index)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
                start_predictions, state, self.take_search_step)
        return {
            "predicted_log_probs": log_probabilities,
            "predictions": all_top_k_predictions
        }
    
    def take_search_step(self,
                         last_predictions: torch.Tensor,
                         state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        用于beam_search。

        参数：
            - last_predictions：上一时间步的预测结果
                    shape: (group_size,)
            - state：状态
        
        返回：
            - final_log_probs：在全词表上的对数似然
                    shape: (group_size, target_vocab_size)
            - state：更新后的状态

        说明：该函数用于提供给Beam Search使用，输入为上一个时间步的预测id（last_predictions，
              初始为start_index），输出为全词表上的对数似然概率（final_log_probs）。
        
        TODO: 考虑OOV情况（需要整体大改）
        """
        # 更新一步decoder状态
        state = self._decoder_step(last_predictions, state)
        
        # 对第一个source的拷贝概率值，shape: (group_size, seq_max_len_1)
        copy_scores_1 = state["attentive_weights_1"]
        # 对第二个source的拷贝概率值，shape: (group_size, seq_max_len_2)
        copy_scores_2 = state["attentive_weights_2"]
        # 概率值的门控，shape: (group_size,)
        gate_score = state["gate_score"]

        # 计算全词表上的对数似然
        final_log_probs = self._gather_final_log_probs(copy_scores_1, copy_scores_2, gate_score, state)

        return final_log_probs, state

    def _gather_final_log_probs(self,
                                copy_scores_1: torch.Tensor,
                                copy_scores_2: torch.Tensor,
                                gate_score: torch.Tensor,
                                state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        根据三个概率，计算全词表上的对数似然。

        参数：
            - copy_scores_1：第一个source的复制概率（经过归一化）
                    shape: (group_size, seq_max_len_1)
            - copy_scores_2：第二个source的复制概率（经过归一化）
                    shape: (group_size, seq_max_len_2)
            - gate_score：门控的分数，决定source1共享多少比例（source2即贡献1-gate_score）
                    shape: (group_size,)
            - state：当前时间步，更新后的解码状态
        
        返回：
            - final_log_probs：全词表上的概率
                    shape: (group_size, target_vocab_size)
        """
        # 获取group_size和两个序列的长度
        group_size, seq_max_len_1 = copy_scores_1.size()
        group_size, seq_max_len_2 = copy_scores_2.size()

        # TODO: 这里默认了source和target使用同一个词表映射，否则需要source2target的映射
        #      （即source词在target词表的index），才能进行匹配
        # shape: (group_size, seq_max_len_1)
        source_token_ids_1 = state["source_token_ids_1"]
        # shape: (group_size, seq_max_len_2)
        source_token_ids_2 = state["source_token_ids_2"]

        # 在序列上扩展gate_score
        # 需要和source1相乘的gate概率，shape: (group_size, seq_max_len_1)
        gate_1 = gate_score.expand(seq_max_len_1, -1).t()
        # 需要和source2相乘的gate概率，shape: (group_size, seq_max_len_2)
        gate_2 = (1 - gate_score).expand(seq_max_len_2, -1).t()

        # 加权后的source1分值，shape: (group_size, seq_max_len_1)
        copy_scores_1 = copy_scores_1 * gate_1
        # 加权后的source2分值，shape: (group_size, seq_max_len_2)
        copy_scores_2 = copy_scores_2 * gate_2

        # shape: (group_size, seq_max_len_1)
        log_probs_1 = (copy_scores_1 + 1e-45).log()
        # shape: (group_size, seq_max_len_2)
        log_probs_2 = (copy_scores_2 + 1e-45).log()
        
        # 初始化全词表上的概率为全0, shape: (group_size, target_vocab_size)
        final_log_probs = (state["decoder_hidden"].new_zeros((group_size,
                self._target_vocab_size)) + 1e-45).log()

        for i in range(seq_max_len_1):  # 遍历source1的所有时间步
            # 当前时间步的预测概率，shape: (group_size, 1)
            log_probs_slice = log_probs_1[:, i].unsqueeze(-1)
            # 当前时间步的token ids，shape: (group_size, 1)
            source_to_target_slice = source_token_ids_1[:, i].unsqueeze(-1)

            # 选出要更新位置，原有的词表概率，shape: (group_size, 1)
            selected_log_probs = final_log_probs.gather(-1, source_to_target_slice)
            # 更新后的概率值（原有概率+更新概率，混合），shape: (group_size, 1)
            combined_scores = util.logsumexp(torch.cat((selected_log_probs,
                    log_probs_slice), dim=-1)).unsqueeze(-1)
            # 将combined_scores设置回final_log_probs中
            final_log_probs = final_log_probs.scatter(-1, source_to_target_slice, combined_scores)
        
        # 对source2也同样做一遍
        for i in range(seq_max_len_2):
            log_probs_slice = log_probs_2[:, i].unsqueeze(-1)
            source_to_target_slice = source_token_ids_2[:, i].unsqueeze(-1)
            selected_log_probs = final_log_probs.gather(-1, source_to_target_slice)
            combined_scores = util.logsumexp(torch.cat((selected_log_probs,
                    log_probs_slice), dim=-1)).unsqueeze(-1)
            final_log_probs = final_log_probs.scatter(-1, source_to_target_slice, combined_scores)
        
        return final_log_probs

    def _get_predicted_tokens(self,
                              predicted_indices: Union[torch.Tensor, numpy.ndarray],
                              batch_metadata: List[Any],
                              n_best: int = None) -> List[Union[List[List[str]], List[str]]]:
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        predicted_tokens: List[Union[List[List[str]], List[str]]] = []
        for top_k_predictions, metadata in zip(predicted_indices, batch_metadata):
            batch_predicted_tokens: List[List[str]] = []
            for indices in top_k_predictions[:n_best]:
                tokens: List[str] = []
                indices = list(indices)
                if self._end_index in indices:
                    indices = indices[:indices.index(self._end_index)]
                for index in indices:
                    token = self.vocab.get_token_from_index(index, self._target_namespace)
                    tokens.append(token)
                batch_predicted_tokens.append(tokens)
            if n_best == 1:
                predicted_tokens.append(batch_predicted_tokens[0])
            else:
                predicted_tokens.append(batch_predicted_tokens)
        return predicted_tokens

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        将预测结果（tensor）解码成token序列。
        """
        predicted_tokens = self._get_predicted_tokens(output_dict["predictions"],
                                                      output_dict["metadata"])
        output_dict["predicted_tokens"] = predicted_tokens
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(self._tensor_based_metric.get_metric(reset=reset))
        return all_metrics