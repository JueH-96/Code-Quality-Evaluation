def main():
    import sys
    x = sys.stdin.read().strip()
    # Remove trailing zeros and possibly a trailing dot if present
    formatted = x.rstrip("0").rstrip(".")
    # Handle the special case when the input is something like "0.000"
    if formatted == "":
        formatted = "0"
    print(formatted)

if __name__ == '__main__':
    main()