class Solution:
    def smallestString(self, s: str) -> str:
        # Find the first character that is not 'a'
        i = 0
        while i < len(s) and s[i] == 'a':
            i += 1

        # If all characters are 'a', replace the last character with 'z'
        if i == len(s):
            return s[:-1] + 'z'

        # Replace the first non-'a' character with 'a'
        return s[:i] + 'a' + s[i+1:]