class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        
        def cost_to_make_equal(start, end):
            # Calculate the cost to make all elements in nums[start:end] equal
            subarray = nums[start:end]
            subarray.sort()
            median = subarray[len(subarray) // 2]  # Optimal value to minimize sum of absolute differences
            return sum(abs(num - median) for num in subarray)
        
        from functools import lru_cache
        
        @lru_cache(None)
        def solve(i, j):
            # Returns the minimum cost to place exactly j subarrays starting from index i
            if j == 0:
                return 0  # No subarrays needed
            if i + j * x > n:
                return float('inf')  # Not enough elements left
            
            # Option 1: Don't place a subarray starting at i
            res = solve(i + 1, j)
            
            # Option 2: Place a subarray starting at i
            if i + x <= n:
                cost = cost_to_make_equal(i, i + x)
                res = min(res, cost + solve(i + x, j - 1))
            
            return res
        
        return solve(0, k)