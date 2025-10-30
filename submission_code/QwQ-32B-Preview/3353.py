class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        if len(s) < 2:
            return False
        rev_s = s[::-1]
        original_substrings = {s[i:i+2] for i in range(len(s)-1)}
        rev_substrings = {rev_s[i:i+2] for i in range(len(rev_s)-1)}
        return not original_substrings.isdisjoint(rev_substrings)