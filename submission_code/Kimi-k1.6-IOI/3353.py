class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # Generate all possible 2-length substrings from the original string
        substrings = [s[i:i+2] for i in range(len(s) - 1)]
        # Reverse the original string
        reversed_s = s[::-1]
        # Check each substring against the reversed string
        for substr in substrings:
            if substr in reversed_s:
                return True
        return False