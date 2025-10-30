def main():
    import sys

    S = sys.stdin.readline().strip()
    n = len(S)

    best = 1  # At least one character is always a palindrome

    # Helper: expand around center
    def expand(l, r):
        nonlocal best
        while l >= 0 and r < n and S[l] == S[r]:
            best = max(best, r - l + 1)
            l -= 1
            r += 1

    for i in range(n):
        # Odd-length palindrome (center at i)
        expand(i, i)
        # Even-length palindrome (center between i and i+1)
        if i + 1 < n:
            expand(i, i + 1)

    print(best)


if __name__ == "__main__":
    main()