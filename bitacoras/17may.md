# Bitácora Técnica — call me maybe
## Sesión 1 — Fundamentos y Diseño Conceptual

---

## 1. ¿Qué ve el LLM cuando le pasas texto?

El modelo **nunca ve texto**. Lo que recibe es una lista de enteros:

```
"What is the sum of 40 and 2?"
        ↓  tokenización
[892, 318, 262, 4771, 286, 2319, 290, 362, 30]
```

El tokenizador convierte cada trozo de texto (subpalabra, puntuación, espacio) en un ID numérico. El modelo opera exclusivamente sobre esos números a través de aritmética matricial. No hay comprensión, hay matemáticas.

---

## 2. Qué son los Logits (y por qué NO son probabilidades)

En cada paso de generación, el modelo produce un vector de puntuaciones brutas llamadas **logits**, uno por cada token del vocabulario (~150.000 en Qwen3).

```
token "{"     →  8.2   ← logit (puntuación bruta)
token "hello" →  1.1
token "42"    →  3.7
token "}"     → -2.4
```

Los logits pueden ser cualquier número real, incluso negativos. **No suman 1. No son probabilidades.**

Para convertirlos en probabilidades se aplica **softmax**:

```
P(token_i) = e^(logit_i) / suma_de_todos(e^(logit_j))
```

Resultado:
```
P("{")     = 0.94
P("hello") = 0.001
P("42")    = 0.05
P("}")     = 0.00
```

Las probabilidades suman 1 siempre, por construcción matemática.

---

## 3. La clave de Constrained Decoding

Si pones el logit de un token a `-infinito` **antes** de aplicar softmax:

```
e^(-infinito) = 0
```

Ese token desaparece del denominador. Su probabilidad es exactamente **0**. El modelo no puede elegirlo bajo ningún concepto.

**Este es el mecanismo completo de constrained decoding:**

```
1. El modelo produce logits para todos los tokens posibles
2. Tu código identifica qué tokens son inválidos en este momento
3. Pones sus logits a -inf
4. Softmax los convierte en probabilidad 0
5. El modelo solo puede elegir tokens válidos
```

Por qué es mejor que hacer prompting pidiendo JSON:
- Un modelo de 0.6B genera JSON válido espontáneamente ~30% de las veces
- Con constrained decoding: 100% siempre, por construcción matemática
- No dependes de que el modelo "quiera" cooperar

---

## 4. Generación Autoregresiva

El proceso se repite **token a token**:

```
Prompt → Tokenizar → Input IDs → LLM → Logits → [CONSTRAINT] → Token elegido
                                                                      ↓
                                              Se añade al prompt y se repite
```

Cada token generado se concatena al input y se vuelve a pasar al modelo. Así hasta que el JSON está completo.

---

## 5. Estructura del JSON de salida (fija)

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "fn_name": "fn_add_numbers",
  "args": {"a": 2.0, "b": 3.0}
}
```

**Parte rígida** — siempre igual, conocida de antemano:
```
{ "fn_name": ... , "args": { ... } }
```

**Parte dinámica** — decidida por el LLM:
- Qué función elegir → `fn_name`
- Qué valores para cada argumento → `args`

---

## 6. Gramática de JSON — Reglas Críticas

Las claves de un objeto JSON **siempre son strings**. Siempre.

```
Después de {
  "clave"  ← ✓ string
  }        ← ✓ objeto vacío
  42       ← ✗ número no puede ser clave
  true     ← ✗ booleano no puede ser clave
```

Los valores válidos dependen del contexto:
- Después de `{"fn_name":` → solo strings (nombres de función)
- Después de `{"args": {"a":` → depende del tipo del parámetro

**Conclusión de diseño:** Los tokens válidos cambian en cada paso según lo que se haya generado antes. Necesitas una máquina de estados.

---

## 7. Estrategia para fn_name: Prefix Matching

`fn_name` es un **conjunto cerrado** — los nombres de función vienen de `function_definitions.json` y son finitos y conocidos de antemano.

La estrategia es **prefix matching**:

```
string acumulado: ""
  → tokens válidos: los que son prefijo de alguna función

string acumulado: "fn_add"
  → tokens válidos: solo los que continúan "fn_add_numbers"
  → "fn_reverse_string" ya no es posible

string acumulado: "fn_add_numbers"
  → match exacto → siguiente token: solo "\"" de cierre
```

**Importante:** Los nombres de función no son un token único. Se tokenizan en partes:
```
"fn_add_numbers" → ["fn", "_add", "_numbers"]
```

Por eso el matching es token a token, acumulando el string parcial.

**No asumir que todas las funciones empiezan por `fn_`.** El evaluador puede cambiar los archivos de entrada. No hardcodear nada.

---

## 8. Estructura de Datos: Diccionario de Prefijos

Para evitar iterar 150.000 tokens en cada paso (demasiado lento), se precalcula **una sola vez** al inicio:

```python
prefijo_a_tokens_validos = {
    "":         {token_id_1, token_id_2, ...},  # tokens que arrancan alguna función
    "fn":       {token_id_X, token_id_Y},        # tokens que continúan alguna función
    "fn_add":   {token_id_Z},                    # solo continúa fn_add_numbers
    ...
}
```

Cómo construirlo:
1. Cargar `function_definitions.json` → nombres de función
2. Cargar el vocabulario con `get_path_to_vocabulary_json()` → mapa `token_id → string`
3. Para cada función, para cada posible prefijo, registrar qué token IDs son válidos

Se construye **antes del bucle de generación**. Durante la generación solo se consulta.

---

## 9. Estrategia para args: Validación por Tipo

Los valores de argumentos **no son un conjunto cerrado**. No puedes precalcularlos.

Lo que sí sabes de antemano: el **tipo** del argumento, desde `function_definitions.json`:

```json
"parameters": {
    "a": {"type": "number"},
    "s": {"type": "string"}
}
```

Para un `number`, los tokens válidos son los que respetan la sintaxis numérica:
```
dígitos 0-9    ← siempre válidos dentro del número
"."            ← válido una sola vez (decimales)
","            ← termina el número, pasa al siguiente arg
"}"            ← termina el objeto args
letras, etc.   ← siempre inválidos dentro de un número
```

Para un `string`, cualquier carácter es válido dentro de las comillas, excepto `"` sin escapar que cierra el string.

La validación la hace **tu código Python**, no el LLM. El LLM solo genera tokens.

---

## 10. La Máquina de Estados

Tu sistema necesita saber en todo momento en qué parte del JSON está. Estados propuestos:

```
EXPECT_FN_NAME    → tokens válidos: nombres de función (prefix matching)
EXPECT_ARGS_KEY   → tokens válidos: nombres de parámetros de la función elegida
EXPECT_ARG_VALUE  → tokens válidos: depende del tipo del parámetro actual
EXPECT_SEPARATOR  → token único válido (: o , o { o })
DONE              → generación completa
```

Transiciones:
```
inicio          → EXPECT_SEPARATOR (el "{" de apertura)
después de "{"  → EXPECT_FN_NAME
fn_name listo   → EXPECT_SEPARATOR (":")
después de ":"  → generando valor de fn_name
fn_name cerrado → EXPECT_SEPARATOR (",")
después de ","  → EXPECT_SEPARATOR ("args")
...
EXPECT_ARGS_KEY → por cada argumento de la función
EXPECT_ARG_VALUE → tipo consultado desde definitions.json
```

---

## 11. El Contexto de Generación (clase propia)

La máquina de estados necesita datos acumulados para funcionar. Esto vive en una clase:

**Responsabilidades:**
- Mantener el estado actual (`EXPECT_FN_NAME`, etc.)
- Guardar qué función eligió el LLM (para consultar sus parámetros)
- Guardar qué argumento se está generando ahora (para saber el tipo)
- Acumular el string parcial del token en construcción
- Responder a la pregunta: **"¿Qué token IDs son válidos ahora mismo?"**

El bucle de generación no sabe nada de JSON. Solo pregunta a esta clase qué tokens son válidos y aplica el constraint.

---

## 12. El Bucle de Generación (pseudocódigo conceptual)

```
inicializar máquina de estados
precalcular diccionario de prefijos
construir prompt inicial

repetir hasta DONE:
    llamar SDK → get_logits_from_input_ids(input_ids_actuales)
    pedir a máquina_de_estados → tokens_validos
    poner a -inf todos los logits excepto tokens_validos
    elegir token con mayor logit restante (argmax)
    añadir token al input_ids acumulado
    actualizar máquina_de_estados con el token elegido
    si máquina_de_estados.estado == DONE: parar

decodificar output → JSON final
validar con pydantic
escribir en output file
```

---

## 13. Arquitectura Modular Propuesta

```
src/
├── __main__.py          → entry point, args parsing
├── loader.py            → leer y validar archivos de entrada
├── prompt_builder.py    → construir el prompt para el LLM
├── vocabulary.py        → cargar vocabulario, construir diccionario de prefijos
├── state_machine.py     → máquina de estados + contexto de generación
├── generator.py         → bucle de generación + aplicar constraints
├── validator.py         → validación pydantic del output
└── writer.py            → escribir JSON de salida
```

Cada módulo tiene **una sola responsabilidad**. El generador no sabe de JSON. El state machine no sabe del LLM. La separación es lo que hace el sistema mantenible.

---

## 14. Decisiones de Diseño Clave y sus Porqués

| Decisión | Alternativa descartada | Por qué |
|----------|----------------------|---------|
| Precalcular diccionario de prefijos | Iterar 150k tokens en cada paso | Tiempo límite de 5 min del subject |
| Máquina de estados explícita | Variables sueltas | Mantenibilidad, claridad de responsabilidades |
| Validación por tipo para args | Precalcular todos los valores posibles | Imposible para números/strings arbitrarios |
| Constraint antes de softmax | Filtrar después | Después ya no puedes eliminar tokens |
| Función elegida por LLM via constraint | Heurísticas (if "sum" in prompt) | El subject lo prohíbe explícitamente |

---

## 15. Lo que queda por diseñar (próxima sesión)

- Detalles del bucle de generación token a token
- Cómo construir exactamente el prompt inicial (qué context dar al LLM)
- Manejo de edge cases: JSON de entrada inválido, función no reconocida, args faltantes
- Integración con pydantic para validación del output
- Estructura de `pyproject.toml` con uv

---

*Bitácora generada tras sesión nocturna de diseño conceptual. El código viene después de que esto esté sólido.*