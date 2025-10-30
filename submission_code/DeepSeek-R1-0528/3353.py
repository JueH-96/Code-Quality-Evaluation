class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        n = len(s)
        if n < 2:
            return False
        rev_s = s[::-1]
        rev_substrings = set()
        for i in range(len(rev_s) - 1):
            rev_substrings.add(rev_s[i:i+2])
        
        for i in range(n - 1):
            if s[i:i+2] in rev_substrings:
                return True
        return False