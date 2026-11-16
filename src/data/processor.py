"""
DataProcessor — Preprocesamiento unificado ES/PT + tokenización XLM-R.

Responsabilidad:
  - Normalización de texto (NFKC, espacios múltiples, strip)
  - Tokenización con XLM-R (padding, truncation a max_length)
  - Procesamiento individual y por batches

Dependencias:
  - transformers.AutoTokenizer
"""

from typing import List
import re
import unicodedata

from transformers import AutoTokenizer


class DataProcessor:
    """Procesador de texto que normaliza y tokeniza usando XLM-R."""

    def __init__(self, model_name: str = "xlm-roberta-base", max_length: int = 128):
        """
        Inicializa el tokenizer de XLM-R.

        Args:
            model_name: Nombre del modelo HF para el tokenizer.
            max_length: Longitud máxima de tokenización (padding/truncation).
        """
        self.max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # XLM-R no tiene pad_token por defecto; usamos eos_token como fallback
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normaliza un texto: NFKC, elimina espacios múltiples, strip.

        Args:
            text: Texto de entrada.

        Returns:
            Texto normalizado.
        """
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def process(self, text: str) -> dict:
        """
        Procesa un texto individual: normaliza y tokeniza.

        Args:
            text: Texto de entrada.

        Returns:
            Dict tokenizado con input_ids, attention_mask, etc.
        """
        normalized = self.normalize_text(text)
        tokenized = self._tokenizer(
            normalized,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,  # return Python lists/dicts
        )
        return tokenized

    def process_batch(self, texts: List[str], batch_size: int = 16) -> List[dict]:
        """
        Procesa una lista de textos en batches: normaliza cada texto y tokeniza por batch.

        Args:
            texts: Lista de textos a procesar.
            batch_size: Tamaño del batch para tokenización.

        Returns:
            Lista de dicts tokenizados.
        """
        results: List[dict] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            normalized_batch = [self.normalize_text(t) for t in batch_texts]

            tokenized = self._tokenizer(
                normalized_batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )

            # Convertir de dict of lists a list of dicts
            for j in range(len(normalized_batch)):
                results.append(
                    {k: tokenized[k][j] for k in tokenized.keys()}
                )

        return results

    def get_tokenizer(self) -> AutoTokenizer:
        """Retorna el tokenizer interno."""
        return self._tokenizer

