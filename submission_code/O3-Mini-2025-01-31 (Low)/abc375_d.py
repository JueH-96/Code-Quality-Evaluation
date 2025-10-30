def main():
    import sys
    input = sys.stdin.readline
    
    S = input().strip()
    n = len(S)
    
    # Precompute prefix counts: prefix[i][c] = number of occurrences of c in S[:i]
    # We'll use list of 26 integers for each position.
    prefix = [[0] * 26 for _ in range(n + 1)]
    for i in range(1, n + 1):
        # copy previous prefix counts
        for j in range(26):
            prefix[i][j] = prefix[i - 1][j]
        # update count for S[i-1]
        prefix[i][ord(S[i - 1]) - ord('A')] += 1
    
    # Similarly precompute suffix counts: suffix[i][c] = number of occurrences of c in S[i-1:]
    suffix = [[0] * 26 for _ in range(n + 2)]
    for i in range(n, 0, -1):
        for j in range(26):
            suffix[i][j] = suffix[i + 1][j]
        suffix[i][ord(S[i - 1]) - ord('A')] += 1
    
    ans = 0
    # Now iterate over j from 2 to n-1 (1-indexed)
    for j in range(2, n):
        # for each letter c, count pairs (i, k) with i < j and j < k such that S[i] and S[k] are letter c.
        for c in range(26):
            ans += prefix[j - 1][c] * suffix[j + 1][c]
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()