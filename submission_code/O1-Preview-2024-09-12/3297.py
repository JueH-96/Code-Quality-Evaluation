class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        for t in range(1, n + 1):
            shift = (k * t) % n
            len_needed = n - shift
            needed_suffix = word[:len_needed]
            available_prefix = word[shift:]
            if needed_suffix == available_prefix:
                return t
        return n