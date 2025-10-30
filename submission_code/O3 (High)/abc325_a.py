def main():
    import sys

    # Read all input and split by whitespace.
    tokens = sys.stdin.read().strip().split()

    if not tokens:
        return  # No input provided (edge‐case safeguard).

    surname = tokens[0]  # The first token is the surname (S).
    # We don't actually need the given first name (T) for the required output.

    print(f"{surname} san")


if __name__ == "__main__":
    main()