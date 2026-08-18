# Bitácora Técnica — call me maybe
## Sesión 3 — Implementación: input/, schemas/, prompt_builder/, llm/

---

## 1. Estructura final del proyecto

```
CALL ME MAYBE/
├── bitacoras/
├── data/
│   ├── input/
│   │   ├── function_definitions.json
│   │   └── function_calling_tests.json
│   └── output/
├── llm_sdk/              ← proporcionado por 42, no se toca
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── input/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── schemasclasses.py
│   ├── prompt_builder/
│   │   ├── __init__.py
│   │   └── builder.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── wrapper.py
│   │   └── vocabulary.py    ← pendiente
│   ├── decoding/            ← pendiente
│   ├── generation/          ← pendiente
│   ├── validation/          ← pendiente
│   └── output/              ← pendiente
├── pyproject.toml
└── README.md
```

---

## 2. input/loader.py

### Decisiones de diseño
- Dos funciones separadas: una por tipo de archivo
- Devuelven datos crudos — la validación pydantic ocurre en `schemas/`
- Los errores se re-lanzan con mensajes descriptivos (no se silencian)
- `__main__.py` decide qué hacer con los errores, no `input/`

### Errores manejados
```
archivo no existe         → FileNotFoundError  (re-lanzado con mensaje)
JSON malformado           → ValueError         (envuelve json.JSONDecodeError)
JSON válido pero no list  → ValueError         (comprobación explícita)
elementos no son del tipo esperado → ValueError (all() + isinstance())
```

### Por qué re-lanzar en vez de capturar
Si `input/` captura y devuelve `None`, `__main__.py` recibe `None` sin saber por qué.
Si `input/` re-lanza con mensaje claro, `__main__.py` recibe información accionable.
Fallar ruidosamente es mejor ingeniería que fallar silenciosamente.

### Funciones
```python
def load_function_definitions(path: str) -> list[dict]
def load_prompts(path: str) -> list[str]
```

### Check de elementos en load_prompts
```python
if not all(isinstance(prompt, str) for prompt in data_prompts):
    raise ValueError("All prompts must be strings.")
```
`all()` con generator expression — para en el primer elemento que falla.
Eficiente, idiomático en Python.

---

## 3. schemas/schemasclasses.py

### Modelos Pydantic

```python
class ParameterDefinition(BaseModel):
    type: str    # "number", "string", "boolean"

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition

class FunctionCallResult(BaseModel):
    prompt: str
    fn_name: str
    args: dict[str, float | str | bool]
```

### Distinción importante
```
schemas/     → validación estructural y tipada (pydantic estático)
validation/  → validación semántica y contextual (runtime)
```

`schemas/` no sabe que `fn_reverse_string` requiere `{"s": string}`.
Esa relación contextual requiere consultar `function_definitions.json` en runtime.
Por eso existe `validation/` como módulo separado.

### Límite de pydantic
```json
{"fn_name": "fn_reverse_string", "args": {"a": 40}}
```
Pydantic acepta esto — `40` es `float`, está en `float | str | bool`.
Pero semánticamente es incorrecto — `fn_reverse_string` requiere `{"s": string}`.
Ese error lo detecta `validation/`, no `schemas/`.

### Si validation/ detecta ese error
Significa que `decoding/` tiene un bug — no cumplió su contrato.
La respuesta correcta es fallar con mensaje claro, no reintentar.
Reintentar sobre un bug solo produce el mismo resultado roto.

---

## 4. prompt_builder/builder.py

### Función
```python
def build_prompt(prompt: str, functions: list[FunctionDefinition]) -> str
```

### Estructura del prompt generado
```
[INSTRUCCIÓN]    → rol y tarea del LLM
[FUNCIONES]      → cada función con nombre, descripción y parámetros
[REQUEST]        → la petición original del usuario
```

### Instrucción al LLM
```
"You are an expert API router. Given the user's request and the list 
of JSON function schemas, select the most accurate function to resolve 
the request and return the selected function name and its required 
parameters strictly in JSON format."
```

### Por qué incluir parámetros en el prompt
```
nombre + description  → Tarea 1: elegir función correcta
parameters            → Tarea 2: extraer argumentos correctos
```
Sin parámetros, el LLM sabe qué función usar pero no qué extraer del prompt.

### Formato de cada función en el prompt
```
fn_add_numbers: Add two numbers
  - a: number
  - b: number
```

### Patrón acumulador
```python
functions_text = ""          # inicializado vacío
for function in functions:
    functions_text += ...    # acumula cada función
```
Sin el `""` inicial, Python lanzaría `NameError` en el primer `+=`.

### Unión final
```python
return "\n\n".join([instruction, functions_text, user_request])
```

---

## 5. llm/wrapper.py

### Principio fundamental
Si el SDK cambia `encode(text)` por `tokenize(text)`, solo cambia `llm/wrapper.py`.
El resto del sistema no sabe que existe el SDK.

### Clase LLMError
```python
class LLMError(Exception):
    pass
```
Error propio del módulo. El resto del sistema solo conoce `LLMError`,
nunca las excepciones internas del SDK.

### Clase LLM
```python
class LLM:
    def __init__(self) -> None:
        self._model = Small_LLM_Model()
```
La instancia se crea una sola vez. Cargar un modelo de 0.6B parámetros
tarda varios segundos — no se puede hacer en cada llamada.

### Métodos de inferencia

```python
def encode(self, text: str) -> list[int]
def decode(self, input_ids: list[int]) -> str
def get_logits(self, input_ids: list[int]) -> np.ndarray
```

### Transformaciones en encode
```
str
 │ self._model.encode(text)
 ▼
Tensor [1, seq_len]
 │ squeeze(0)
 ▼
Tensor [seq_len]
 │ tolist()
 ▼
list[int]
```

### Transformaciones en get_logits
El SDK acepta `list[int]` directamente y devuelve `list[float]`.
```python
logits = self._model.get_logits_from_input_ids(input_ids)
return np.array(logits)
```
No se necesita torch — el SDK ya maneja la conversión internamente.

### Conexión token_id ↔ logits (crítico para decoding/)
```
logits[token_id] = puntuación de ese token
```
El vector tiene exactamente el mismo tamaño que el vocabulario.
El índice ES el token ID.

```
logits[0]      → puntuación del token con ID 0
logits[5423]   → puntuación del token "fn"
logits[149999] → puntuación del último token
```

### Patrón de manejo de errores en todos los métodos
```python
try:
    ...
except Exception as e:
    raise LLMError(f"mensaje descriptivo: {e}") from e
```
`from e` mantiene el traceback original para debugging.

---

## 6. llm/vocabulary.py (pendiente de implementar)

### Por qué un objeto Vocabulary y no un dict
Exponer `dict` directamente acopla toda la aplicación a una estructura concreta.
Si mañana cambias la implementación interna, tienes que tocar todos los módulos.

Exponer un objeto con métodos es más robusto:
```python
vocabulary.get_token_string(token_id)
vocabulary.get_ids_for_prefix(prefix)
```
La implementación interna puede cambiar y el resto del sistema no se entera.

### Métodos que necesitará decoding/
```
¿Qué string corresponde a este token_id?   → para prefix matching
¿Qué token_ids empiezan por este prefijo?  → para constrained decoding
```

### Dónde vive
`llm/vocabulary.py` — el vocabulario viene del SDK, pertenece a `llm/`.
`decoding/` recibe un objeto `Vocabulary` ya construido, no sabe de rutas ni JSON.

---

## 7. Lo que queda por implementar

```
□ llm/vocabulary.py      → objeto Vocabulary con métodos de consulta
□ decoding/              → máquina de estados + constrained decoding
□ generation/            → bucle autoregresivo
□ validation/            → validación semántica con FunctionDefinition
□ output/                → serialización JSON final
□ __main__.py            → orquestador + CLI
□ Makefile               → install, run, debug, clean, lint
□ README.md              → documentación completa
```

---

*Sesión 3 completada. Siguiente: llm/vocabulary.py y decoding/.*