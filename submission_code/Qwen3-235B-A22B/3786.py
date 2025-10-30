class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        if n == 0:
            return 0
        dp = [[dict() for _ in range(n)] for __ in range(n)]

        # Initialize base cases
        for i in range(n):
            dp[i][i][1] = 0

        # Fill DP table
        for length in range(2, n + 1):  # current substring length
            for i in range(n - length + 1):
                j = i + length - 1
                # compute cost_ij
                a = s[i]
                b = s[j]
                diff = abs(ord(a) - ord(b))
                cost_ij = min(diff, 26 - diff)

                # case3: combine i and j
                temp = {}
                inner = dp[i + 1][j - 1]
                # iterate inner items:
                for l_inner in inner:
                    c_inner = inner[l_inner]
                    new_l = l_inner + 2
                    new_cost = c_inner + cost_ij
                    if new_l not in temp or new_cost < temp.get(new_l, float('inf')):
                        temp[new_l] = new_cost

                # handle adding the case where the inner interval is invalid.
                if (i + 1 > j - 1):
                    new_l = 2
                    new_cost = cost_ij
                    if new_l not in temp or new_cost < temp.get(new_l, float('inf')):
                        temp[new_l] = new_cost

                # Now merge case1, case2, case3 into candidates
                # case1: dp[i+1][j]
                case1 = dp[i + 1][j] if (i + 1 <= j) else {}
                # case2: dp[i][j-1]
                case2 = dp[i][j - 1] if (j - 1 >= i) else {}

                candidates = {}

                # add case1 entries
                for l in case1:
                    c = case1[l]
                    if c < candidates.get(l, float('inf')):
                        candidates[l] = c
                # add case2 entries
                for l in case2:
                    c = case2[l]
                    if c < candidates.get(l, float('inf')):
                        candidates[l] = c
                # add temp entries
                for l in temp:
                    c = temp[l]
                    if c < candidates.get(l, float('inf')):
                        candidates[l] = c

                # update dp[i][j]
                dp[i][j] = candidates.copy()

        # Now, check dp[0][n-1]
        max_len = 0
        for l in dp[0][n - 1]:
            if l > max_len and dp[0][n - 1][l] <= k:
                max_len = l
        return max_len