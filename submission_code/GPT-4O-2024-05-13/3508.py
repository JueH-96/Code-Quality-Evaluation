class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # If k has bits set to 1 that are not set in n, it's impossible
        if k & ~n:
            return -1
        
        # Count the number of 1s in n and k
        n_ones = bin(n).count('1')
        k_ones = bin(k).count('1')
        
        # The number of changes needed is the difference in the number of 1s
        return n_ones - k_ones

# Example usage:
# sol = Solution()
# print(sol.minChanges(13, 4))  # Output: 2
# print(sol.minChanges(21, 21))  # Output: 0
# print(sol.minChanges(14, 13))  # Output: -1