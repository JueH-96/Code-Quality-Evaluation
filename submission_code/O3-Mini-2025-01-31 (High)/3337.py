class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        # Count the occurrences of character c in s
        count = s.count(c)
        # The total number of substrings that start and end with c is given by:
        # count choose 2 (for substrings with length > 1) plus count (for single character substrings)
        return count * (count + 1) // 2