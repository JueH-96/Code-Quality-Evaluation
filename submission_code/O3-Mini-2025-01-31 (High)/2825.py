class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # The key observation is that the operation allows us to remove duplicate occurrences 
        # of any character, while always leaving one instance of each character.
        # Therefore, in the minimized string, each character appears at most once.
        # In other words, the minimized length is simply the number of distinct characters in s.
        return len(set(s))