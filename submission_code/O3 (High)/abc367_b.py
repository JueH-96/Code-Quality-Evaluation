def main():
    import sys

    # Read the input as a raw string and strip the newline
    s = sys.stdin.readline().strip()

    # Remove trailing zeros from the fractional part,
    # then remove a trailing decimal point if it remains.
    trimmed = s.rstrip('0').rstrip('.')

    # In the unlikely event the result is empty (e.g., input "0"),
    # fall back to printing "0".
    if trimmed == "":
        trimmed = "0"

    print(trimmed)


if __name__ == "__main__":
    main()