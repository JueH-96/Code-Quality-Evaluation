class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Find the position of the '*' in the pattern
        star_index = p.index('*')

        # Split the pattern into two parts: before and after the '*'
        prefix = p[:star_index]
        suffix = p[star_index + 1:]

        # Check if the prefix and suffix can be found in the string s
        for i in range(len(s) - len(prefix) - len(suffix) + 1):
            if s[i:i + len(prefix)] == prefix and s[i + len(prefix):i + len(prefix) + len(suffix)] == suffix:
                return True

        return False