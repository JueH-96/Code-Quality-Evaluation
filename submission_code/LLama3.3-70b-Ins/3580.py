class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        def is_almost_equal(s: str, pattern: str) -> bool:
            """Check if two strings are almost equal."""
            diff_count = 0
            for i in range(len(pattern)):
                if s[i] != pattern[i]:
                    diff_count += 1
                if diff_count > 1:
                    return False
            return True

        pattern_len = len(pattern)
        min_index = -1
        for i in range(len(s) - pattern_len + 1):
            # Check if the substring is almost equal to the pattern
            if is_almost_equal(s[i:i + pattern_len], pattern):
                # Update the minimum index
                if min_index == -1 or i < min_index:
                    min_index = i

        return min_index