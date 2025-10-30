def main():
    import sys

    # Read A and B
    data = sys.stdin.read().strip().split()
    if not data:
        return
    A, B = map(int, data)

    # Sum
    s = A + B

    # Pick a number in 0..9 different from s
    # (s + 1) % 10 is always within range and never equals s
    ans = (s + 1) % 10

    print(ans)


if __name__ == "__main__":
    main()