# Bitácora Técnica — call me maybe
## Sesión 4 — Decoding: Estado, Vocabulario y Esqueleto del Decoder

---

## 1. Problemas de entorno resueltos

### El problema de torch y CUDA
El `llm_sdk` proporcionado por 42 tenía `torch 2.13.0+cu130` como dependencia,
compilado para CUDA 13. La máquina de 42 no tiene CUDA 13 instalado a nivel de sistema.

**Solución:** Copiar el `uv.lock` del compañero que usa `torch 2.10.0` (manylinux),
que bundlea sus propias librerías sin depender del sistema.

```
Tu lockfile:   torch 2.13.0+cu130   ← necesita libcupti.so.13 del sistema
Su lockfile:   torch 2.10.0          ← manylinux, compatible sin CUDA instalado
```

### El problema del espacio en disco
El home de 42 está al 100%. El modelo Qwen3-0.6B (1.5GB) intenta descargarse ahí.

**Solución permanente:**
```bash
echo 'export HF_HOME=/goinfre/$USER/.cache/huggingface' >> ~/.zshrc
source ~/.zshrc
```

Así el modelo se descarga en goinfre (195G libres).

### Estructura verificada del vocabulario
```python
vocab = json.load(f)
# resultado: {token_string: token_id}
# ejemplo: [('!', 0), ('"', 1), ('#', 2), ...]
```

La clase `Vocabulary` construye internamente `_id_to_token` invirtiendo el diccionario.

### Retorno de encode verificado
El SDK devuelve `torch.Tensor` con forma `[1, seq_len]`:
```python
def encode(self, text: str) -> torch.Tensor:
    ids = self._tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor([ids], device=self._device, dtype=torch.long)
```

Por eso el wrapper hace `.squeeze(0).tolist()` — correcto.

---

## 2. llm/wrapper.py — versión final

```
LLM
├── __init__  → instancia Small_LLM_Model una sola vez
├── encode    → str → list[int]  (squeeze + tolist)
├── decode    → list[int] → str
└── get_logits → list[int] → np.ndarray  (np.array directo, SDK devuelve list[float])
```

Error propio: `LLMError(Exception)` — el resto del sistema nunca ve excepciones del SDK.

---

## 3. llm/vocabulary.py — versión final

### Decisión de diseño
No exponer `dict` directamente. Exponer un objeto con métodos:

```python
vocabulary.get_token(token_id)   # int → str
vocabulary.get_token_id(token)   # str → int
len(vocabulary)                  # tamaño del vocabulario
```

### Por qué
Si mañana cambia la implementación interna, el resto del sistema no se entera.
Exponer un `dict` acopla toda la aplicación a la estructura concreta.

### Conexión crítica con logits
```
logits[token_id] = puntuación de ese token
```
El vector de logits tiene exactamente el mismo tamaño que el vocabulario.
El índice ES el token ID. Esta relación es fundamental para decoding.

---

## 4. Estructura de src/decoding/

```
src/decoding/
├── __init__.py
├── state.py      → DecodingState (Enum) + DecoderState (BaseModel)
└── decoder.py    → apply_constraints() + update_state()
```

---

## 5. state.py — diseño completo

### DecodingState (Enum)
Conjunto cerrado de fases. No permite estados arbitrarios.

```python
class DecodingState(Enum):
    EXPECT_FN_NAME    = "expect_fn_name"
    EXPECT_ARGS_KEY   = "expect_args_key"
    EXPECT_ARGS_VALUE = "expect_args_value"
    EXPECT_SEPARATOR  = "expect_separator"
    DONE              = "done"
```

### DecoderState (BaseModel)
Contexto dinámico que acompaña al estado.

```python
class DecoderState(BaseModel):
    phase: DecodingState
    selected_function: str | None = None
    current_parameter: str | None = None
    prefix: str = ""
```

### Separación conceptual
```
DecodingState  →  "¿En qué fase de la máquina estoy?"
DecoderState   →  "¿Qué información concreta tengo en esa fase?"
```

### Estado inicial
```python
DecoderState(phase=DecodingState.EXPECT_FN_NAME)
# phase = EXPECT_FN_NAME
# selected_function = None
# current_parameter = None
# prefix = ""
```

### Transiciones de la máquina
```
             START
               │
               ▼
       EXPECT_FN_NAME
               │
        nombre completo
               │
               ▼
       EXPECT_ARGS_KEY
               │
         key encontrada
               │
               ▼
      EXPECT_ARGS_VALUE
               │
       valor completo
               │
               ▼
       EXPECT_SEPARATOR
               │
        más args? ── sí ──► EXPECT_ARGS_KEY
               │
               no
               ▼
              DONE
```

---

## 6. decoder.py — esqueleto y decisiones de diseño

### Por qué dos funciones separadas

```
apply_constraints()  →  PREPARA la decisión  (antes del argmax)
update_state()       →  REGISTRA la decisión (después del argmax)
```

Son dos momentos distintos del bucle:

```
LLM.get_logits()
        ↓
apply_constraints(logits, state, functions, vocabulary)
        ↓
logits filtrados
        ↓
argmax()
        ↓
token_id
        ↓
update_state(token_id, state, functions, vocabulary)
        ↓
nuevo state
        ↓
¿DONE? → parar o repetir
```

### Por qué NO juntar todo en apply_constraints
No puedes actualizar correctamente el estado antes de conocer el token seleccionado.
Durante EXPECT_FN_NAME con prefix="fn_a", varios tokens podrían ser válidos
("dd_numbers", "dd_strings"...). Hasta que argmax elija uno, no sabes qué
transición corresponde. Juntarlo mezcla dos momentos distintos del algoritmo.

### Ventajas de la separación
- `apply_constraints` es pura respecto al estado: consulta, no modifica
- `update_state` trabaja con token ya elegido, no mezcla predicción con transición
- Testeable por separado
- Si un token inválido sobrevivió → problema en `apply_constraints`
- Si el estado avanzó mal → problema en `update_state`

### Firmas finales

```python
def apply_constraints(
    logits: np.ndarray,
    state: DecoderState,
    functions: list[FunctionDefinition],
    vocabulary: Vocabulary,
) -> np.ndarray:
    """Aplica restricciones de decodificación a los logits."""
    pass


def update_state(
    token_id: int,
    state: DecoderState,
    functions: list[FunctionDefinition],
    vocabulary: Vocabulary,
) -> DecoderState:
    """Actualiza el estado tras seleccionar un token."""
    pass
```

---

## 7. Lo que queda por implementar

```
□ apply_constraints()  → lógica de prefix matching y filtrado de logits
□ update_state()       → lógica de transiciones de estado
□ generation/          → bucle autoregresivo completo
□ validation/          → validación semántica
□ output/              → serialización JSON final
□ __main__.py          → orquestador + CLI
□ Makefile             → install, run, debug, clean, lint
□ README.md            → documentación completa
```

### Próximo paso concreto
Implementar `apply_constraints()` empezando por el caso más simple:
`EXPECT_FN_NAME` con prefix matching contra los nombres de función disponibles.

---

*Sesión 4 completada. El esqueleto de decoding está listo. Siguiente: lógica interna de apply_constraints.*