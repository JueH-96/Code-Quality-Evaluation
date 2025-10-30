class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        def count_ways(remaining, max_base):
            if remaining == 0:
                return 1
            if remaining < 0 or max_base == 0:
                return 0
            if (remaining, max_base) in memo:
                return memo[(remaining, max_base)]
            
            take = count_ways(remaining - max_base**x, max_base - 1)
            skip = count_ways(remaining, max_base - 1)
            
            memo[(remaining, max_base)] = (take + skip) % MOD
            return memo[(remaining, max_base)]
        
        # Calculate the maximum base that can be used for the given n and x
        max_base = int(n**(1/x))
        memo = {}
        return count_ways(n, max_base)

# Example usage:
# sol = Solution()
# print(sol.numberOfWays(10, 2))  # Output: 1
# print(sol.numberOfWays(4, 1))   # Output: 2