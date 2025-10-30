class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Check if s can be transformed into t by rotating
        if s == t:
            # If k is even, we can always rotate the string back to itself in 2 moves
            # If k is odd, we can only do it if n is even (by rotating half the string)
            return (k + (n % 2 == 0)) % MOD
        
        # Check if s can be transformed into t by rotating any number of times
        if s not in (t * 2):
            return 0
        
        # Find the index where s needs to be split to match t after rotation
        split_index = (t + t).find(s)
        
        # If k is 1, there's only one way to transform s into t
        if k == 1:
            return 1
        
        # If k is greater than 1, we can either directly rotate to match t,
        # or rotate to a different string and then back to t in the remaining moves
        # There are two ways to rotate back to t: either from the left or the right
        # However, we need to exclude the direct rotation, which we already counted
        return (2 * (k - 1) - (split_index == 0 or split_index == n)) % MOD

# Example usage:
sol = Solution()
print(sol.numberOfWays("abcd", "cdab", 2))  # Output: 2
print(sol.numberOfWays("ababab", "ababab", 1))  # Output: 2