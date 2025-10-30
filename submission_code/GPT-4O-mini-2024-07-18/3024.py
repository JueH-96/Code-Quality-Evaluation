class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)

        # Check if s can be transformed into t by rotation
        if sorted(s) != sorted(t):
            return 0

        # Find all valid rotations of s that match t
        valid_rotations = 0
        for i in range(n):
            if s[i:] + s[:i] == t:
                valid_rotations += 1

        # Calculate the number of ways to perform k operations
        # Each valid rotation can be performed in k ways
        # The result is valid_rotations * (k // n + 1) if k >= n
        # Otherwise, it's valid_rotations * (k // n)
        if k >= n:
            return (valid_rotations * (k // n + 1)) % MOD
        else:
            return (valid_rotations * (k // n)) % MOD