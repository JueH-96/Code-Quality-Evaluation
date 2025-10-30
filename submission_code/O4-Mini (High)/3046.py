class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        INF = float('inf')
        ans = INF
        # Try all multiples-of-25 endings of length 2
        for pat in ("00", "25", "50", "75"):
            c1, c2 = pat[0], pat[1]
            # find c2 as the last character
            for j in range(n - 1, -1, -1):
                if num[j] != c2:
                    continue
                # find c1 before position j
                for i in range(j - 1, -1, -1):
                    if num[i] == c1:
                        # deletions between i and j, and after j
                        ops = (j - 1 - i) + (n - 1 - j)
                        ans = min(ans, ops)
                        break  # no need to look for an earlier i for this j
        # Also consider leaving a single '0' (which is divisible by 25)
        if '0' in num:
            ans = min(ans, n - 1)
        # If still INF, must delete all digits to get "0"
        if ans == INF:
            return n
        return ans