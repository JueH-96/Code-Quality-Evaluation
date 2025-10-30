class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        max_t = n // k
        for t in range(1, max_t + 1):
            start = t * k
            suffix = word[start:]
            if word.startswith(suffix):
                return t
        # If no valid t found in the loop, the next possible t is when the entire word is rotated, which is ceil(n/k)
        # However, according to problem constraints and examples, this case should not occur
        return (n + k - 1) // k