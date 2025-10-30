class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        if k == 1 or k == len(word):
            return 1
        
        # The pattern repeats every len(word) / gcd(len(word), k) times
        from math import gcd
        return len(word) // gcd(len(word), k)