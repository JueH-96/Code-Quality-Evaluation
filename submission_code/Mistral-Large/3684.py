class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split the pattern into two parts: before and after the '*'
        star_index = p.index('*')
        prefix = p[:star_index]
        suffix = p[star_index + 1:]

        # Check if the prefix matches the start of the string
        if not s.startswith(prefix):
            return False

        # Check if the suffix matches the end of the string
        if not s.endswith(suffix):
            return False

        # Check if the remaining part of the string (after prefix and before suffix)
        # can be replaced by the '*'
        remaining_length = len(s) - len(prefix) - len(suffix)
        return remaining_length >= 0