def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    seen = set()
    # Enumerate all non-empty substrings
    for i in range(n):
        for j in range(i + 1, n + 1):
            seen.add(S[i:j])
    print(len(seen))

if __name__ == "__main__":
    main()