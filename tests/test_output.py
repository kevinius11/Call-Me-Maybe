from src.output import save_results
from src.schemas import FunctionCallResult


def test_save_results() -> None:
    results = [
        FunctionCallResult(
            prompt="What is the sum of 2 and 3?",
            fn_name="fn_add_numbers",
            args={
                "a": 2,
                "b": 3,
            },
        ),
    ]

    path = "data/output/test_output.json"

    save_results(results, path)

    with open(path, encoding="utf-8") as file:
        content = file.read()

    expected = """[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": {
      "a": 2,
      "b": 3
    }
  }
]"""

    assert content == expected

    print("output: OK")


if __name__ == "__main__":
    test_save_results()
