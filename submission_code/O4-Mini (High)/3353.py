class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        """
        Check if there exists any substring of length 2 in s that also appears
        in the reverse of s.
        """
        rev_s = s[::-1]
        # Iterate through all substrings of length 2 in s
        for i in range(len(s) - 1):
            sub = s[i:i+2]
            if sub in rev_s:
                return True
        return False