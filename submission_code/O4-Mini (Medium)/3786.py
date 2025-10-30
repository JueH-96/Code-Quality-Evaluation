class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        # dp optimization: dp_next[j][t] = max pairs in s[i+1..j] using exactly t ops
        # we'll roll i from n-1 down to 0
        # initialize dp_next for i = n (i+1 > j): zero
        dp_next = [[0] * (k + 1) for _ in range(n)]
        # helper to compute circular distance between two chars
        def cost_match(a, b):
            diff = abs(ord(a) - ord(b))
            return diff if diff <= 26 - diff else 26 - diff

        # iterate i from n-1 down to 0
        for i in range(n - 1, -1, -1):
            dp_cur = [None] * n
            # j == i: empty substring or single char, no pairs
            dp_cur[i] = [0] * (k + 1)
            for j in range(i + 1, n):
                c1 = s[i]
                c2 = s[j]
                cost = cost_match(c1, c2)
                dp1 = dp_next[j]      # skip i
                dp2 = dp_cur[j - 1]   # skip j
                dp3 = dp_next[j - 1]  # pair i and j
                # build dp_cur[j]
                row = [0] * (k + 1)
                # for each possible op count t
                # maximize skip transitions first
                for t in range(k + 1):
                    # skip i or skip j
                    m = dp1[t] if dp1[t] >= dp2[t] else dp2[t]
                    # try pairing if enough ops
                    if t >= cost:
                        # dp3[t - cost] may be smaller, add 1
                        val = dp3[t - cost] + 1
                        if val > m:
                            m = val
                    row[t] = m
                dp_cur[j] = row
            # roll
            dp_next = dp_cur

        # dp_next[n-1][t] is max pairs with exactly t ops
        max_pairs = max(dp_next[n - 1]) if n > 0 else 0
        # total length = 2*pairs, plus 1 if there's at least one unused character
        # i.e., if 2*pairs < n
        res = 2 * max_pairs + (1 if 2 * max_pairs < n else 0)
        return res