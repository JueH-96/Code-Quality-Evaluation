class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # Get the length of the string
        n = len(s)

        # Iterate through all substrings of length 2
        for i in range(n - 1):
            substring = s[i:i+2]
            # Check if the substring is present in the reverse of s
            if substring in s[::-1]:
                return True

        # If no such substring is found, return False
        return False