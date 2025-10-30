class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # The key observation is that, for each distinct letter,
        # we can remove all but one occurrence using the operation.
        # Thus, the minimal string contains at most one occurrence per distinct letter.
        # So, the answer is the number of unique characters in s.
        return len(set(s))