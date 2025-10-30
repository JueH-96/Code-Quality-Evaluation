class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        # prefix sums for zeros and ones
        pref0 = [0] * (n + 1)
        pref1 = [0] * (n + 1)
        for i, ch in enumerate(s):
            pref0[i + 1] = pref0[i] + (1 if ch == '0' else 0)
            pref1[i + 1] = pref1[i] + (1 if ch == '1' else 0)
        
        ans = 0
        # check every substring s[i:j]
        for i in range(n):
            for j in range(i, n):
                zeros = pref0[j + 1] - pref0[i]
                ones = pref1[j + 1] - pref1[i]
                # satisfies if zeros <= k or ones <= k
                if zeros <= k or ones <= k:
                    ans += 1
        return ans