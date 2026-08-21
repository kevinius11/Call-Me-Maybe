# Bitácora - 21 de agosto

## LLM wrapper

Se consolidó `src/llm/model.py` como wrapper del SDK `Small_LLM_Model`.

La clase `LLM` expone:

- `encode(text: str) -> list[int]`
- `decode(token_ids: list[int]) -> str`
- `get_logits(input_ids: list[int]) -> np.ndarray`

Se creó `LLMError` para traducir errores procedentes del SDK.

Se verificó directamente el SDK y se confirmó que:

- `encode()` devuelve un tensor con dimensión batch `[1, seq_len]`.
- `get_logits_from_input_ids()` devuelve `list[float]`, por lo que el wrapper convierte directamente esa salida a `np.ndarray`.
- El SDK proporciona acceso público al vocabulario mediante una ruta al archivo JSON.

## Vocabulary

Se creó `src/llm/vocabulary.py`.

La clase `Vocabulary` carga el vocabulario JSON, cuyo formato es:

```text
token -> token_id