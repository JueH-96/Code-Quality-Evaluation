class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        # Precompute character codes
        codes = [ord(ch) for ch in s]
        # dp_nxt[j][c] = max palindromic subseq length in s[i+1..j] with ≤c operations
        # Initialize for "i = n" (empty substring): all zeros
        dp_nxt = [[0] * (k + 1) for _ in range(n)]
        
        # Fill dp for i from n-1 down to 0
        for i in range(n - 1, -1, -1):
            dp_cur = [None] * n
            for j in range(i, n):
                if i == j:
                    # Single char: can always take it as palindrome of length 1
                    dp_cur[j] = [1] * (k + 1)
                else:
                    top = dp_nxt[j]       # dp[i+1][j]
                    left = dp_cur[j - 1]  # dp[i][j-1]
                    diag = dp_nxt[j - 1]  # dp[i+1][j-1]
                    # cost to match s[i] and s[j] into the same letter
                    d = abs(codes[i] - codes[j])
                    cost_pair = d if d <= 26 - d else 26 - d
                    
                    # build dp_cur[j] for costs 0..k
                    row = [0] * (k + 1)
                    if cost_pair == 0:
                        # free to match (letters already equal)
                        for c in range(k + 1):
                            skip_val = top[c] if top[c] >= left[c] else left[c]
                            pair_val = diag[c] + 2
                            row[c] = pair_val if pair_val > skip_val else skip_val
                    else:
                        # for c < cost_pair: can't match, only skip
                        for c in range(cost_pair):
                            v1 = top[c]
                            v2 = left[c]
                            row[c] = v1 if v1 >= v2 else v2
                        # for c >= cost_pair: choose skip or match-pair
                        for c in range(cost_pair, k + 1):
                            # skip option
                            v1 = top[c]
                            v2 = left[c]
                            skip_val = v1 if v1 >= v2 else v2
                            # pair option (use c - cost_pair ops inside)
                            pair_val = diag[c - cost_pair] + 2
                            row[c] = pair_val if pair_val > skip_val else skip_val
                    dp_cur[j] = row
            # move current row to "next" for the next iteration of i
            dp_nxt = dp_cur
        
        # result is the best we can do on entire string s[0..n-1] with ≤ k ops
        return max(dp_nxt[n - 1])