class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        reversed_s = s[::-1]
        substrs = set()
        n = len(s)
        for i in range(n - 1):
            substr = s[i] + s[i+1]
            substrs.add(substr)
        for substr in substrs:
            if substr in reversed_s:
                return True
        return False