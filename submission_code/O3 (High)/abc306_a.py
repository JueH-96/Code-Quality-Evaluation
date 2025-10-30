def main():
    import sys

    # Read input
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    N = int(data[0])           # We don't actually need N besides reading it
    S = data[1] if len(data) > 1 else ""

    # Build the result by doubling each character
    doubled = ''.join(c * 2 for c in S)

    # Output the result
    print(doubled)


if __name__ == "__main__":
    main()