class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        if len(s) < 2:
            return False
        substrings = {s[i:i+2] for i in range(len(s)-1)}
        reversed_substrings = {s[::-1][i:i+2] for i in range(len(s)-1)}
        return bool(substrings & reversed_substrings)