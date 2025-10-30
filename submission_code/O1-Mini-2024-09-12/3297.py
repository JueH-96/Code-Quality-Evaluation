import math

class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        # Calculate the minimum t using the formula ceil((n - k + 1) / k)
        t = math.ceil((n - k + 1) / k)
        return t if t > 0 else 1