class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Check if t is a rotation of s
        if t not in (s + s):
            return 0
        
        # Find the number of positions where s can be rotated to get t
        rotation_count = 0
        for i in range(n):
            if s[i:] + s[:i] == t:
                rotation_count += 1
        
        # Calculate the number of ways to achieve exactly k operations
        if k == 0:
            return 1 if s == t else 0
        
        # Calculate the number of valid rotations in k operations
        result = 0
        for i in range(k // n + 1):
            if (k - i * n) % rotation_count == 0:
                result += 1
        
        return result % MOD