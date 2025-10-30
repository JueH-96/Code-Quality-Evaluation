class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        reversed_s = s[::-1]
        substrings = set()

        # Collect all substrings of length 2 from the original string
        for i in range(len(s) - 1):
            substr = s[i:i+2]
            substrings.add(substr)
        
        # Collect all substrings of length 2 from the reversed string
        for i in range(len(reversed_s) - 1):
            if reversed_s[i:i+2] in substrings:
                return True
        return False