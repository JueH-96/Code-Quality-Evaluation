class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        n, m = len(s), len(t)
        # 1) Compute all palindromic substrings in s
        pal_s = [[False]*n for _ in range(n)]
        best_pal_s = 0
        for i in range(n-1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i < 2 or pal_s[i+1][j-1]):
                    pal_s[i][j] = True
                    best_pal_s = max(best_pal_s, j - i + 1)
        # 2) Compute all palindromic substrings in t
        pal_t = [[False]*m for _ in range(m)]
        best_pal_t = 0
        for i in range(m-1, -1, -1):
            for j in range(i, m):
                if t[i] == t[j] and (j - i < 2 or pal_t[i+1][j-1]):
                    pal_t[i][j] = True
                    best_pal_t = max(best_pal_t, j - i + 1)
        # 3) For s: best_len_s_l[i] = max length of a palindrome substring starting at i
        best_len_s_l = [0]*n
        for i in range(n):
            mx = 0
            for j in range(i, n):
                if pal_s[i][j]:
                    length = j - i + 1
                    if length > mx:
                        mx = length
            best_len_s_l[i] = mx
        # 4) For t: best_len_t_r[j] = max length of a palindrome substring ending at j
        best_len_t_r = [0]*m
        for j in range(m):
            mx = 0
            for i in range(j+1):
                if pal_t[i][j]:
                    length = j - i + 1
                    if length > mx:
                        mx = length
            best_len_t_r[j] = mx
        # 5) Compute K_max[i][j]: the longest prefix match of s[i:] with reverse of t[:j+1]
        K_max = [[0]*m for _ in range(n)]
        for i in range(n-1, -1, -1):
            for j in range(m):
                if s[i] == t[j]:
                    if i+1 < n and j-1 >= 0:
                        K_max[i][j] = 1 + K_max[i+1][j-1]
                    else:
                        K_max[i][j] = 1
        # 6) Initialize answer by the best single‐string palindrome
        ans = max(best_pal_s, best_pal_t)
        # 7) Try cross‐boundary palindromes
        for i in range(n):
            for j in range(m):
                K = K_max[i][j]
                if K <= 0:
                    continue
                # k = length of the matching "core" between s[i..] and reverse(t[..j])
                for k in range(1, K+1):
                    # Option 1: extend on the s side
                    extra_s = best_len_s_l[i+k] if (i+k < n) else 0
                    cand1 = 2*k + extra_s
                    if cand1 > ans:
                        ans = cand1
                    # Option 2: extend on the t side
                    extra_t = best_len_t_r[j-k] if (j-k >= 0) else 0
                    cand2 = 2*k + extra_t
                    if cand2 > ans:
                        ans = cand2
        return ans