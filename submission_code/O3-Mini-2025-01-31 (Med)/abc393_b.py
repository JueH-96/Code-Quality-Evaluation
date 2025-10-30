def main():
    import sys
    input_str = sys.stdin.read().strip()
    if not input_str:
        return
    S = input_str.strip()
    n = len(S)
    count = 0
    # Since indexes in the problem are 1-indexed, but Python is 0-indexed, adjust indexing accordingly.
    # We need to find triples (i, j, k) with 1 <= i < j < k <= n, j-i = k-j, S[i-1]=='A', S[j-1]=='B', S[k-1]=='C'
    # We can choose j (the middle index) from 2 to n-1 (1-indexed) 
    for j in range(2, n):  # j is the 1-indexed middle position
        d = 1
        # i = j - d must be at least 1 and k = j + d at most n
        while (j - d >= 1) and (j + d <= n):
            if S[j - d - 1] == 'A' and S[j - 1] == 'B' and S[j + d - 1] == 'C':
                count += 1
            d += 1
    sys.stdout.write(str(count))

if __name__ == '__main__':
    main()