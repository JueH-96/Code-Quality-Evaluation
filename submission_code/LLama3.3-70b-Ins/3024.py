class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Calculate the number of rotations that result in the same string
        rotations = 0
        for i in range(1, n):
            if s[i:] + s[:i] == s:
                rotations += 1
        
        # If s is already equal to t, we can perform any number of rotations
        if s == t:
            if k == 0:
                return 1
            elif k == 1:
                return rotations + 1
            else:
                return (rotations + 1) * pow(rotations, k-1, MOD) % MOD
        
        # If s is not equal to t, we need to find the number of ways to transform s into t
        # We can do this by checking all possible suffixes of s and appending them to the start of s
        ways = 0
        for i in range(1, n):
            new_s = s[i:] + s[:i]
            if new_s == t:
                ways += 1
        
        # If we can transform s into t in one operation, we can do it in any number of operations
        if ways > 0:
            if k == 1:
                return ways
            else:
                return ways * pow(rotations + 1, k-1, MOD) % MOD
        
        # If we cannot transform s into t in one operation, we cannot do it in any number of operations
        return 0