from typing import Dict, Optional, Iterable

from overrides import overrides
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.tokenizers import Token
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


@DatasetReader.register("ms")
class MSDatasetReader(DatasetReader):
    def __init__(self,
                 namespace: str = "tokens",
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source1_token_indexers = {"tokens": SingleIdTokenIndexer(namespace=namespace)}
        self._source2_token_indexers = {"tokens": SingleIdTokenIndexer(namespace=namespace)}
        self._target_token_indexers = {"tokens": SingleIdTokenIndexer(namespace=namespace)}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            src1, src2, tgt = line.split("\t")
            src1, src2, tgt = src1.strip(), src2.strip(), tgt.strip()
            if not src1 or not src2 or not tgt:
                continue
            yield self.text_to_instance(src1, src2, tgt)
        
    @overrides
    def text_to_instance(self,
                         source1: str,
                         source2: str,
                         target: Optional[str] = None) -> Instance:
        # 对source1和source2分词、添加END
        source1_tokens = [Token(token) for token in source1.split(" ")]
        source2_tokens = [Token(token) for token in source2.split(" ")]
        source1_tokens.append(Token(END_SYMBOL))
        source2_tokens.append(Token(END_SYMBOL))
        source1_field = TextField(source1_tokens, self._source1_token_indexers)
        source2_field = TextField(source2_tokens, self._source2_token_indexers)

        meta_fields = {
            "source_tokens_1": [x.text for x in source1_tokens[:-1]],
            "source_tokens_2": [x.text for x in source2_tokens[:-1]]
        }
        fields_dict = {
            "source_tokens_1": source1_field,
            "source_tokens_2": source2_field
        }

        if target is not None:
            # 对target分词、添加START、END
            assert all(any(tgt_token == src_token for src_token in source1.split(" ")) or
                       any(tgt_token == src_token for src_token in source2.split(" "))
                       for tgt_token in target.split(" ")), "target词必须在两个source中出现"
            target_tokens = [Token(token) for token in target.split(" ")]
            target_tokens.insert(0, Token(START_SYMBOL))
            target_tokens.append(Token(END_SYMBOL))
            target_field = TextField(target_tokens, self._target_token_indexers)

            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in target_tokens[1:-1]]
        
        fields_dict["metadata"] = MetadataField(meta_fields)
        return Instance(fields_dict)