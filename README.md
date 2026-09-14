*This project has been created as part of the 42 curriculum by <kcastro->.*

# Call-Me-Maybe

## Description

Call-Me-Maybe is a function-calling system built around a small language model
(Qwen/Qwen3-0.6B) and a constrained decoding mechanism.

The goal of the project is to transform a natural-language user request into a
structured JSON function call. The generated call must contain:

- the name of one of the available functions;
- the parameters required by that function;
- values whose types match the function definition.

The project does not rely only on the language model to produce valid JSON.
Instead, it combines semantic prompting with a constrained decoder based on a
finite state machine (FSM).

The main pipeline is:

```text
Natural-language prompt
        ↓
Prompt builder
        ↓
Qwen/Qwen3-0.6B
        ↓
Constrained decoding
        ↓
Generated JSON
        ↓
Semantic validation
        ↓
Pydantic model
        ↓
JSON output
````

The function definitions are loaded from a JSON file and represented using
Pydantic models. The decoder uses those definitions to restrict the tokens that
can be generated at each step.

---

## Features

* Function selection from natural-language prompts.
* Parameter extraction from user requests.
* Constrained autoregressive decoding.
* Finite-state-machine based decoding.
* Token-level vocabulary filtering.
* Number and string validation during generation.
* Function and parameter validation after generation.
* Pydantic-based data models.
* Automatic creation of the output directory.
* Command-line support for custom input and output files.
* Linting and type checking through the Makefile.

---

## Instructions

### Requirements

The project requires:

* Python 3.13 or newer.
* `uv`.

The project uses the provided `llm_sdk` and the Qwen/Qwen3-0.6B model.

### Installation

Install and synchronize the project dependencies with:

```bash
make install
```

This is equivalent to:

```bash
uv sync
```

### Running the project

The default command is:

```bash
make run
```

or:

```bash
uv run python -m src
```

By default, the program expects:

```text
data/input/function_calling_tests.json
data/input/function_definitions.json
```

and writes the final results to:

```text
data/output/function_calling_results.json
```

The output directory is created automatically when it does not already exist.

### Custom input and output

The input prompts and output path can be changed from the command line:

```bash
uv run python -m src \
    --input path/to/input.json \
    --output path/to/output.json
```

The function definitions are loaded from:

```text
data/input/function_definitions.json
```

### Debugging

The Makefile provides a debug rule:

```bash
make debug
```

### Linting

Run the static checks with:

```bash
make lint
```

This runs:

```text
flake8
mypy
```

### Cleaning temporary Python files

```bash
make clean
```

This removes Python cache directories and compiled Python files.

---

## Algorithm Explanation

The core of the project is a constrained autoregressive decoder.

The language model produces logits for the next token. Instead of selecting
the maximum-logit token directly, the decoder first determines which tokens are
valid according to the current decoding state.

Invalid tokens are assigned a logit of negative infinity:

```text
invalid token → -∞
valid token   → original logit
```

The token with the highest remaining logit is then selected.

### Finite State Machine

The decoder uses four main states:

```text
EXPECT_FN_NAME
EXPECT_ARGS_KEY
EXPECT_ARGS_VALUE
DONE
```

#### 1. EXPECT_FN_NAME

The decoder only allows tokens that can continue one of the function names
defined in the function definitions file.

Once a complete function name has been generated, the closing quote is allowed
and the decoder moves to:

```text
EXPECT_ARGS_KEY
```

#### 2. EXPECT_ARGS_KEY

The decoder obtains the selected function definition and restricts generation
to parameter names belonging to that function.

For example, if the selected function contains:

```json
"parameters": {
    "a": {"type": "number"},
    "b": {"type": "number"}
}
```

the decoder will not allow an unrelated key to be generated.

#### 3. EXPECT_ARGS_VALUE

The decoder looks at the parameter definition and applies type-specific
constraints.

For numeric parameters:

* digits are allowed;
* a negative sign can begin a number;
* one decimal point is allowed;
* the number must be complete before the value can be closed.

For string parameters:

* string content is allowed;
* the closing quote terminates the current value;
* the decoder determines whether another parameter follows.

Regular-expression parameters have additional token restrictions in the
current implementation.

#### 4. DONE

Once the final parameter has been completed, the decoder reaches the `DONE`
state and generation stops.

### Fixed JSON structure

The generator inserts parts of the JSON structure itself instead of asking the
model to generate every syntax token freely.

For example:

```text
{"fn_name": "
```

and later:

```text
, "args": {"
```

are inserted as fixed encoded fragments.

This reduces the search space and makes the state transitions easier to control.

### Generation loop

The generation process is:

```text
1. Build semantic prompt.
2. Encode the prompt.
3. Insert the fixed JSON prefix.
4. Ask the model for logits.
5. Apply decoder constraints.
6. Apply generation bias for string termination.
7. Select the highest valid logit.
8. Update the decoder state.
9. Insert any required fixed JSON syntax.
10. Repeat until DONE.
```

A maximum generation limit is also enforced to prevent an invalid generation
loop from running indefinitely.

---

## Design Decisions

### Finite-state constrained decoding

A finite-state machine was chosen because the output format has a small and
well-defined structure.

Each state represents a specific stage of JSON generation, which makes the
constraints explicit and easier to debug.

### Schema-driven parameter validation

The decoder does not hardcode the parameter names of individual functions.
Instead, it obtains the available parameters from the loaded
`FunctionDefinition` models.

This allows the decoder to operate on different function definitions without
changing the implementation.

### Separation of responsibilities

The project is divided into independent modules:

```text
input/
    Loads JSON input files.

schemas/
    Defines validated data models.

llm/
    Wraps the provided language-model SDK.

prompt_builder/
    Builds the semantic prompt.

decoding/
    Implements constrained token generation.

generation/
    Orchestrates model generation and FSM synchronization.

validation/
    Performs semantic validation after generation.

output/
    Writes the final JSON result.
```

This separation prevents the main entry point from containing model logic,
decoding logic, validation logic, or file-writing logic.

### Pydantic

Pydantic is used for structural validation of function definitions and final
function-call results.

This provides an additional validation layer after constrained decoding.

### Greedy decoding

The generator uses the highest valid logit at every step:

```text
next_token = argmax(valid_logits)
```

This keeps the implementation deterministic and simple.

---

## Performance Analysis

The project prioritizes correctness and constrained generation over maximum
generation speed.

### Accuracy

On the provided set of 11 example prompts:

* the expected function was selected for all examples;
* the generated output was valid JSON;
* the required output fields were produced;
* numeric and string argument types were correctly represented.

One semantic limitation remains in the current example set. For the vowel
replacement prompt, the generated values were:

```json
{
  "regex": "aeiouAEIOU",
  "replacement": "asterisks"
}
```

This is structurally valid and corresponds to the requested function, but it is
not the most precise representation of the requested replacement operation.
This demonstrates that constrained decoding guarantees structural correctness,
but does not automatically guarantee perfect semantic interpretation.

### Speed

The model-loading phase is the dominant startup cost. The generation itself
requires repeated model inference because logits are requested for each
generated step.

The constrained decoder also iterates through the available vocabulary to
determine which tokens remain valid.

The project therefore favors predictable constrained generation and correctness
over raw throughput.

### Reliability

Reliability is improved through multiple validation layers:

```text
Input validation
      ↓
Pydantic function definitions
      ↓
Constrained decoding
      ↓
JSON parsing
      ↓
Semantic validation
      ↓
Pydantic output model
      ↓
JSON writer
```

The decoder also has a maximum token limit to prevent non-terminating
generation.

---

## Challenges Faced

### Constrained generation versus model behavior

One of the main challenges was balancing model behavior with strict decoding
constraints.

Restricting the regex parameter too aggressively caused valid output patterns
to become impossible to generate.

On the other hand, allowing unrestricted regex characters could cause the model
to continue generating for too long.

The final implementation therefore uses a controlled regex constraint together
with a generation bias for string termination.

### Premature string termination

String values can be difficult to terminate reliably because the model must
decide when the generated value is complete.

A small generation bonus is therefore applied to the closing quote during
string generation.

The regex parameter uses a separate value because its generation behavior is
different from normal string parameters.

### Constrained decoding loops

Reducing the regex closing bonus too much caused the model to continue producing
tokens until the maximum generation limit was reached.

This showed that constrained decoding must balance:

```text
valid token freedom
```

with:

```text
reliable termination
```

### Input format

The provided prompts file contains objects with a `prompt` field rather than
raw strings.

The input loader was therefore adapted to validate the file structure and
convert it into the internal `list[str]` representation used by the generator.

### Separation of validation and generation

Another design challenge was deciding where semantic correctness should be
checked.

The decoder is responsible for structural and token-level constraints, while
the `validation` module performs semantic validation after generation.

This keeps the generator independent from the final output serialization.

---

## Testing Strategy

Testing was performed at several levels.

### Decoder tests

The decoder was tested independently to verify:

* valid state transitions;
* invalid tokens being rejected;
* function-name restrictions;
* parameter-name restrictions;
* numeric values;
* string values;
* regex handling;
* completion conditions.

### Validation tests

The semantic validator was tested against:

```text
Valid function calls
Unknown functions
Missing parameters
Extra parameters
Invalid parameter types
Unexpected top-level fields
```

All of these cases are explicitly checked.

### Output tests

The output writer was tested independently to ensure that:

* the output directory is created automatically;
* valid `FunctionCallResult` objects are serialized;
* the generated JSON can be loaded again.


### Invalid Input Testing

The input layer was also tested against malformed and invalid JSON files.

The following cases were verified:

#### Invalid JSON syntax

```bash
uv run python -m src --input /tmp/invalid_syntax.json
````

Example result:

```text
Error: Invalid JSON in /tmp/invalid_syntax.json: Expecting ',' delimiter: line 3 column 1 (char 23)
```

#### Missing input file

```bash
uv run python -m src --input /tmp/no_existe.json
```

Example result:

```text
Error: Prompts file not found: /tmp/no_existe.json
```

#### Invalid JSON structure

A valid JSON object was provided instead of the expected list:

```bash
uv run python -m src --input /tmp/not_a_list.json
```

Example result:

```text
Error: JSON must be a list, not object/dictionaries.
```

#### Invalid prompt entry

A list containing an invalid prompt entry was tested:

```bash
uv run python -m src --input /tmp/bad_items.json
```

Example result:

```text
Error: All prompt entries must be dictionaries.
```

These tests confirm that malformed input files are detected before the generation
pipeline starts and are reported as controlled errors instead of exposing a
Python traceback to the user.

```markdown
#### Invalid function definition

An invalid function definition was tested with a parameter collection that was
not a JSON object.

The Pydantic validation layer correctly rejected the definition before model
generation started.
```

### Integration testing

The complete application was executed using:

```bash
uv run python -m src
```

and the resulting file was checked with:

```bash
cat data/output/function_calling_results.json
```

The output directory was also removed before execution to confirm that it is
recreated automatically.

### Static analysis

The project also uses:

```bash
make lint
```

which runs:

```text
flake8
mypy
```

---

## Example Usage

### Default execution

```bash
make run
```

This reads:

```text
data/input/function_calling_tests.json
data/input/function_definitions.json
```

and produces:

```text
data/output/function_calling_results.json
```

### Custom input, function definitions, and output

The input prompts, function definitions, and output path can be changed from
the command line:

```bash
uv run python -m src \
    --functions_definition data/input/function_definitions.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

### Example prompt

Input:

```text
What is the sum of 2 and 3?
```

Possible generated function call:

```json
{
  "fn_name": "fn_add_numbers",
  "args": {
    "a": 2.0,
    "b": 3.0
  }
}
```

Another example:

```text
Greet john
```

Result:

```json
{
  "fn_name": "fn_greet",
  "args": {
    "name": "john"
  }
}
```

---

## Resources

### Function calling

* OpenAI, Function Calling documentation:
  [https://platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)

* Hugging Face, Text Generation documentation:
  [https://huggingface.co/docs/transformers/main/en/main_classes/text_generation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

### Qwen

* Qwen model family:
  [https://github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen)

* Qwen on Hugging Face:
  [https://huggingface.co/Qwen](https://huggingface.co/Qwen)

### Python

* Python documentation:
  [https://docs.python.org/3/](https://docs.python.org/3/)

* JSON module:
  [https://docs.python.org/3/library/json.html](https://docs.python.org/3/library/json.html)

### Pydantic

* Pydantic documentation:
  [https://docs.pydantic.dev/](https://docs.pydantic.dev/)

### NumPy

* NumPy documentation:
  [https://numpy.org/doc/](https://numpy.org/doc/)

### Regular expressions

* Python `re` documentation:
  [https://docs.python.org/3/library/re.html](https://docs.python.org/3/library/re.html)

---

## AI Usage

AI tools were used as development assistance during the project.

They were mainly used for:

* discussing the project architecture and separation of responsibilities;
* reasoning about finite-state-machine based constrained decoding;
* debugging token-generation and state-transition issues;
* analyzing generation failures and non-terminating decoding;
* reviewing Python code and improving docstrings;
* designing semantic validation and output handling;
* creating and reviewing isolated tests;
* investigating errors related to input parsing, JSON generation, and module
  execution.

AI assistance was also used to discuss alternative implementations and debugging
strategies.

The final implementation, project structure, integration, testing, and code
changes were reviewed and executed in the project environment.

---

## Project Structure

```text
Call-Me-Maybe/
├── data/
│   ├── input/
│   │   ├── function_calling_tests.json
│   │   └── function_definitions.json
│   └── output/
│       └── function_calling_results.json
├── llm_sdk/
├── src/
│   ├── decoding/
│   ├── generation/
│   ├── input/
│   ├── llm/
│   ├── output/
│   ├── prompt_builder/
│   ├── schemas/
│   ├── validation/
│   └── __main__.py
├── tests/
├── bitacoras/
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## Makefile Commands

| Command        | Description                                  |
| -------------- | -------------------------------------------- |
| `make install` | Install and synchronize project dependencies |
| `make run`     | Run the complete function-calling pipeline   |
| `make debug`   | Run the project using Python's debugger      |
| `make lint`    | Run flake8 and mypy                          |
| `make clean`   | Remove Python cache files                    |

````
