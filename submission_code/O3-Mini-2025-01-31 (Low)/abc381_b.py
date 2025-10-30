def main():
    import sys
    S = sys.stdin.read().strip()
    
    # Condition 1: |S| is even
    if len(S) % 2 != 0:
        print("No")
        return

    # Condition 2: For each pair, the two characters are equal.
    n = len(S)
    for i in range(0, n, 2):
        if S[i] != S[i+1]:
            print("No")
            return

    # Condition 3: Each character appears either 0 or 2 times.
    # Because we already have paired equal characters, we can check count by iterating over each pair.
    freq = {}
    for c in S:
        freq[c] = freq.get(c, 0) + 1

    for count in freq.values():
        if count != 2:
            print("No")
            return

    print("Yes")

if __name__ == '__main__':
    main()