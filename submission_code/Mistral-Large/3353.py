class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # Generate the reverse of the string
        reversed_s = s[::-1]

        # Generate all substrings of length 2 from the original string
        substrings = {s[i:i+2] for i in range(len(s) - 1)}

        # Check if any of these substrings are present in the reversed string
        for substring in substrings:
            if substring in reversed_s:
                return True

        return False