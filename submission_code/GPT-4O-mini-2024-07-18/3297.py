class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        # We need to find the smallest t such that after t seconds, the string returns to its original state
        # This happens when the characters shifted out and in align back to the original string
        
        # The effective length of the string after t seconds will be t * k
        # We need to find the smallest t such that (t * k) % n == 0
        # This means t * k is a multiple of n
        
        # We can find the least common multiple (LCM) of k and n
        from math import gcd
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        # Calculate the least common multiple of k and n
        lcm_kn = lcm(k, n)
        
        # The minimum time t is then lcm_kn // k
        return lcm_kn // k