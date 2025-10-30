class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        seen = set()
        for i in range(len(s) - 1):
            substr = s[i:i+2]
            if substr in seen:
                return True
            seen.add(substr[::-1])
        return False