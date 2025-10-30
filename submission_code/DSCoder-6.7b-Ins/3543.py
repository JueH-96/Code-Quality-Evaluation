class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        for length in range(1, n+1):
            for start in range(n-length+1):
                substring = s[start:start+length]
                if substring.count('0') <= k and substring.count('1') <= k:
                    count += 1
        return count