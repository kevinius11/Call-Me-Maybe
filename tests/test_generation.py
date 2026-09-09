from src.llm import LLM


def main() -> None:
    llm = LLM()

    texts = [
        '{"fn_name": "',
        ', "args": {"',
        ': ',
        '"',
        ', "',
        '}',
    ]

    for text in texts:
        token_ids = llm.encode(text)
        decoded = llm.decode(token_ids)

        print("TEXT   :", repr(text))
        print("IDS    :", token_ids)
        print("DECODED:", repr(decoded))
        print("-" * 40)


if __name__ == "__main__":
    main()
