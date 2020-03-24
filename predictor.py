from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('ms_pointer')
class MSPointerPredictor(Predictor):

    def predict(self, source1: str, source2: str) -> JsonDict:
        return self.predict_json({"source1": source1, "source2": source2})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        source1 = json_dict["source1"]
        source2 = json_dict["source2"]
        return self._dataset_reader.text_to_instance(source1, source2)