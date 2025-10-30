def main() -> None:
    import sys

    # Read the string (strip only the trailing newline from stdin)
    s = sys.stdin.readline().rstrip('
')

    # Remove every '.' character
    cleaned = s.replace('.', '')

    # Output the resulting string (even if it's empty, print a newline)
    print(cleaned)


if __name__ == "__main__":
    main()