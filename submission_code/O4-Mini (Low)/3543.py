class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        # prefix sums for zeros and ones
        pref0 = [0] * (n + 1)
        pref1 = [0] * (n + 1)
        for i, ch in enumerate(s):
            pref0[i+1] = pref0[i] + (1 if ch == '0' else 0)
            pref1[i+1] = pref1[i] + (1 if ch == '1' else 0)
        
        count = 0
        # enumerate all substrings s[i:j], inclusive of i, exclusive of j
        for i in range(n):
            for j in range(i+1, n+1):
                zeros = pref0[j] - pref0[i]
                ones = pref1[j] - pref1[i]
                if zeros <= k or ones <= k:
                    count += 1
        return count