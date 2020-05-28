from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class SGNNTokenizer(object):
    def __init__(
        self,
        word_ngram_range: Tuple[int, int] = (1,2),
        char_ngram_range: Tuple[int, int] = (2,3),
        max_features: int = 500,
        ):
        self._word_ngram_range = word_ngram_range
        self._char_ngram_range = char_ngram_range
        self._max_features = max_features
        self._tokenizers = self._create_tokenizers()
    

    def _create_tokenizers(self):
        tokenizers = []
        for i in range(self._word_ngram_range[0], self._word_ngram_range[-1] + 1):
            tokenizers.append(
                CountVectorizer(
                    analyzer='word',
                    ngram_range=(i, i),
                    max_features=self._max_features
                    )
                )
        for i in range(self._char_ngram_range[0], self._char_ngram_range[-1] + 1):
            tokenizers.append(
                CountVectorizer(
                    analyzer='char',
                    ngram_range=(i, i),
                    max_features=self._max_features
                    )
                )
        return tokenizers


    def fit_on_texts(self, texts: List[str]) -> list:
        for tokenizer in self._tokenizers:
            tokenizer.fit(texts)
        return self._tokenizers
    

    def texts_to_matrix(
        self,
        texts: List[str],
        mode: Optional[str] = None
        ) -> Tuple[np.ndarray]:
        matrices = []
        for tokenizer in self._tokenizers:
            matrices.append(tokenizer.transform(texts).toarray())
        return tuple(matrices)
        