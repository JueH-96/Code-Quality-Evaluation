class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # Store all substrings of length 2
        substrings = set()
        for i in range(len(s) - 1):
            substrings.add(s[i:i+2])
        
        # Check if any reversed substring of length 2 exists in our set
        for i in range(len(s) - 1):
            # The reversed substring would be s[i+1] + s[i]
            if s[i+1] + s[i] in substrings:
                return True
        
        return False