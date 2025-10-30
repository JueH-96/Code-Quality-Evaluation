class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Check if t is a rotation of s
        def is_rotation(s, t):
            return t in s + s and len(s) == len(t)
        
        # Count the number of rotations of s that match t
        def count_rotations(s, t):
            return (s + s).count(t) - 1 if s == t else (s + s).count(t)
        
        # Calculate the number of ways to transform s into t in exactly k operations
        def calculate_ways(n, k, rotations):
            if k == 1:
                return rotations
            if rotations == 0:
                return 0
            if rotations == n:
                return (k % 2 == 0) * (n - 1) % MOD
            if rotations == 1:
                return (k % 2 == 0) * (n - 1) % MOD
            return (k % 2 == 0) * (n - 1) % MOD
        
        if not is_rotation(s, t):
            return 0
        
        rotations = count_rotations(s, t)
        return calculate_ways(n, k, rotations)