class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        original_substrings = set()
        n = len(s)
        for i in range(n - 1):
            original_substrings.add(s[i:i+2])
        reversed_s = s[::-1]
        for i in range(len(reversed_s) - 1):
            sub = reversed_s[i:i+2]
            if sub in original_substrings:
                return True
        return False