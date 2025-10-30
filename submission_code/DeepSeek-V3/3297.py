class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        # The number of steps required is the smallest t such that t * k is a multiple of n
        # Because after t steps, the total shift is t * k, and we need this to be a multiple of n
        # So t * k ≡ 0 mod n
        # The smallest t > 0 is the smallest t such that t * k is a multiple of n
        # Which is t = n / gcd(n, k)
        
        from math import gcd
        g = gcd(n, k)
        t = (n // g)
        return t