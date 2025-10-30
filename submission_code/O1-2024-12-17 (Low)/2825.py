class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # The key observation is that, through the described operations,
        # we can eliminate repeated occurrences of a character until only
        # (at most) one instance of each character remains. Therefore, the
        # minimized length is simply the number of distinct characters in s.
        return len(set(s))