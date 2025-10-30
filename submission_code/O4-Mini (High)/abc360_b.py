def main():
    S, T = input().split()
    n = len(S)
    # Try all possible widths w (1 <= w < n)
    for w in range(1, n):
        # For each width, try all possible positions c (1 <= c <= w)
        for c in range(1, w + 1):
            # Build the string by taking the c-th character of each block of size w
            extracted = []
            pos = c - 1
            while pos < n:
                extracted.append(S[pos])
                pos += w
            if ''.join(extracted) == T:
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()