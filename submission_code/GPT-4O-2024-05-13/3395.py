class Solution:
    def minAnagramLength(self, s: str) -> int:
        # The minimum possible length of the string t is the number of unique characters in s
        return len(set(s))