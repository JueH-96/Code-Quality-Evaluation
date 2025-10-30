def main():
    import sys

    S = sys.stdin.readline().strip()
    n = len(S)
    substrings = set()

    # Generate all possible non-empty substrings
    for i in range(n):
        for j in range(i + 1, n + 1):
            substrings.add(S[i:j])

    # The answer is the size of the set of distinct substrings
    print(len(substrings))

if __name__ == "__main__":
    main()