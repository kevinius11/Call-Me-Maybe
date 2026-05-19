# Bitácora Técnica — call me maybe
## Sesión 2 — Arquitectura Modular y Flujo de Datos

---

## 1. Arquitectura Final

```
src/
├── __main__.py      → orquestador, CLI
├── input/           → leer y parsear archivos
├── schemas/         → modelos pydantic
├── prompt_builder/  → construir prompt para el LLM
├── llm/             → wrapper del SDK
├── decoding/        → máquina de estados + constraints
├── generation/      → bucle autoregresivo
├── validation/      → validación semántica
└── output/          → serialización final
```

---

## 2. Decisiones de Diseño y sus Porqués

### ¿Por qué `prompt_builder/` es un módulo propio?

`input/` no debe saber que existe un LLM.
`llm/` no debe saber de funciones ni de prompts de negocio.
`decoding/` controla tokens, no lenguaje natural.

Construir el prompt requiere conocer:
- las funciones disponibles → viene de `input/`
- el request del usuario → viene de `input/`
- cómo formatear para el LLM → conocimiento de prompting

Esa combinación de responsabilidades justifica un módulo propio.

### ¿Por qué `decoding/` aplica el `-inf` y no `generation/`?

`decoding/` tiene toda la información necesaria — conoce los tokens válidos.
La interfaz hacia `generation/` es:

```python
decoding.apply_constraint(logits) → logits modificados
```

`generation/` nunca sabe qué tokens fueron invalidados ni por qué.
Eso es bajo acoplamiento — cada módulo conoce solo lo mínimo necesario.

### ¿Por qué `validation/` usa excepciones y no booleanos?

El subject exige mensajes de error claros al usuario.
Un booleano solo dice "algo salió mal". Una excepción dice exactamente qué:

```
ValidationError: fn_name 'fn_nonexistent' not found in definitions
ValidationError: missing required argument 'b' for fn_add_numbers
```

### ¿`validation/` y `schemas/` se solapan?

No validan el mismo nivel de verdad:

```
schemas/     → validación estructural y tipada (pydantic estático)
validation/  → validación semántica y contextual (runtime)
```

Ejemplo concreto:
```json
{"fn_name": "fn_reverse_string", "args": {"a": 40}}
```

Pydantic valida que existen `fn_name` y `args`.
Pero solo `validation/` sabe que `fn_reverse_string` requiere `{"s": string}`, 
no `{"a": number}`. Eso requiere consultar `function_definitions.json` en runtime.

### ¿Por qué no "resetear" si `validation/` falla?

Si el constrained decoding es correcto, el error semántico es imposible por construcción.
Si `validation/` falla, significa que hay un bug en `decoding/`.

Reintentar silenciosamente es peligroso:
```
bug en decoding/ → reintento → mismo bug → loop infinito
                             → o peor: resultado incorrecto que parece correcto
```

Lo correcto es fallar con mensaje claro. Fallar ruidosamente es mejor ingeniería.

---

## 3. Flujo Completo de Datos

Para el prompt `"What is the sum of 40 and 2?"`:

### `input/`
```
entra:  ruta a function_definitions.json
        ruta a function_calling_tests.json
sale:   lista de FunctionDefinition (pydantic)
        lista de strings (los prompts)

errores manejados:
  archivo no existe          → error claro
  archivo no es JSON válido  → error claro
  estructura incorrecta      → error claro
```

### `schemas/`
```
entra:  datos crudos del JSON
sale:   FunctionDefinition validados con pydantic
        FunctionCallResult (modelo de salida)
        si hay error → ValidationError con mensaje descriptivo
```

### `prompt_builder/`
```
entra:  UN prompt del usuario (string)        ← uno, no la lista
        lista de FunctionDefinition
sale:   string formateado para el LLM

ejemplo de output:
  "Tienes estas funciones disponibles:
   - fn_add_numbers: Add two numbers (params: a: number, b: number)
   - fn_reverse_string: Reverse a string (params: s: string)
   El usuario pide: 'What is the sum of 40 and 2?'
   Elige la función correcta y sus argumentos."
```

El LLM lee `"Add two numbers"` y `"sum of 40 and 2"` y conecta semánticamente.
Sin heurísticas. Sin keywords. Razonamiento real sobre las descripciones.

### `llm/`
```
entra:  input_ids (lista de enteros)
sale:   logits (tensor de puntuaciones)

responsabilidad única: wrapper del SDK
NO sabe qué es FunctionDefinition
NO sabe qué es un prompt de negocio
```

### `generation/`
```
entra:  prompt (string, del prompt_builder/)
        máquina de estados inicializada (de decoding/)
        lista de FunctionDefinition (para que decoding/ sepa qué restringir)
sale:   JSON sin validar (string)

bucle interno:
  1. encode(prompt) → input_ids          ← UNA SOLA VEZ antes del bucle
  2. llm.get_logits(input_ids)           → logits crudos
  3. decoding.apply_constraint(logits)   → logits modificados
  4. argmax(logits modificados)          → siguiente token
  5. añadir token a input_ids
  6. decoding.actualizar_estado(token)
  7. si estado == DONE → parar
  8. repetir desde 2

IMPORTANTE: encode solo ocurre una vez.
Dentro del bucle solo se añaden tokens nuevos, no se re-encodea todo.
```

### `validation/`
```
entra:  FunctionCallResult sin validar (parseado del string)
        lista de FunctionDefinition    ← la fuente de verdad semántica
sale:   FunctionCallResult validado
        o lanza ValidationError con mensaje descriptivo

comprobaciones:
  ¿existe fn_name en las definiciones?         ✓/✗
  ¿tiene los parámetros correctos?             ✓/✗
  ¿los tipos de args coinciden?                ✓/✗
  ¿hay argumentos extra no permitidos?         ✓/✗
```

### `output/`
```
entra:  FunctionCallResult validado
        o ValidationError capturada
sale:   objeto JSON añadido al archivo de salida

caso éxito:
  {"prompt": "...", "fn_name": "...", "args": {...}}

caso error:
  {"prompt": "...", "error": "ValidationError: fn_name 'fn_nonexistent' not found"}

IMPORTANTE: un error en un prompt no para los demás.
El archivo siempre tiene un objeto por prompt.
```

### `__main__.py`
```
responsabilidad: orquestar, nada más

flujo:
  1. parsear argumentos CLI (--input, --output)
  2. llamar a input/
  3. llamar a prompt_builder/
  4. para cada prompt:
       llamar a generation/
       llamar a validation/
       llamar a output/

si __main__.py empieza a crecer →
señal de que lógica de negocio se está colando donde no debe
```

---

## 4. La Máquina de Estados (detalle de `decoding/`)

### Estados
```
EXPECT_FN_NAME    → tokens válidos: nombres de función (prefix matching)
EXPECT_ARGS_KEY   → tokens válidos: nombres de parámetros de la función elegida
EXPECT_ARG_VALUE  → tokens válidos: depende del tipo del parámetro actual
EXPECT_SEPARATOR  → token único válido (: o , o { o })
DONE              → generación completa
```

### Contexto de generación (clase propia)
Datos que la máquina necesita acumular:
```
estado_actual      → EXPECT_FN_NAME, EXPECT_ARG_VALUE...
funcion_actual     → guardado cuando el LLM elige fn_name
argumento_actual   → guardado cuando el LLM elige una key de args
string_parcial     → token en construcción (para prefix matching)
```

Responsabilidad principal: responder a la pregunta
**"¿Qué token IDs son válidos ahora mismo?"**

### Diccionario de prefijos (precalculado UNA SOLA VEZ)
```python
prefijo_a_tokens_validos = {
    "":         {token_ids que arrancan alguna función},
    "fn":       {token_ids que continúan alguna función},
    "fn_add":   {token_id de "_numbers" únicamente},
    ...
}
```

Se construye antes del bucle de generación leyendo:
- nombres de función desde `function_definitions.json`
- vocabulario desde `get_path_to_vocabulary_json()`

Durante la generación solo se consulta, nunca se reconstruye.

### Estrategia por tipo de dato
```
fn_name (conjunto cerrado):
  → prefix matching contra nombres de función conocidos

args keys (conjunto cerrado por función):
  → prefix matching contra parámetros de la función elegida

args values tipo number:
  → validación sintáctica: dígitos, ".", ",", "}"
  → tu código Python, no el LLM

args values tipo string:
  → cualquier carácter hasta comilla de cierre
  → tu código Python, no el LLM
```

---

## 5. Orden de Implementación

Empezar por `input/` porque:
- no depende de ningún otro módulo
- da datos reales para probar el resto
- los errores aquí son fatales
- es el módulo más testeable independientemente

Después: `schemas/` → `prompt_builder/` → `llm/` → `decoding/` → `generation/` → `validation/` → `output/`

---

## 6. Lo que queda por implementar

- Código real de cada módulo
- Tests con edge cases: cadenas vacías, números grandes, caracteres especiales,
  prompts ambiguos, funciones con múltiples parámetros
- Makefile con reglas: install, run, debug, clean, lint
- README.md con todas las secciones requeridas
- pyproject.toml con uv

---

*Sesión 2 completada. Diseño conceptual terminado. Siguiente paso: implementar `input/`.*