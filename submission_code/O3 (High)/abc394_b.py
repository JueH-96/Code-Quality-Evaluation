def main() -> None:
    import sys

    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0])
    strings = data[1:n + 1]            # read exactly N strings
    # Sort by length and concatenate
    result = ''.join(sorted(strings, key=len))
    print(result)


if __name__ == "__main__":
    main()