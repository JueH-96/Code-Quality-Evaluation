class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        n = len(nums)
        max_product = -1
        
        def backtrack(index, count, alt_sum, product):
            nonlocal max_product
            
            # Prune if product exceeds limit
            if product > limit:
                return
            
            # Base case: processed all elements
            if index == n:
                if count > 0 and alt_sum == k:
                    max_product = max(max_product, product)
                return
            
            # Option 1: Skip current element
            backtrack(index + 1, count, alt_sum, product)
            
            # Option 2: Include current element
            if count % 2 == 0:  # Next position is even (0-indexed)
                new_sum = alt_sum + nums[index]
            else:  # Next position is odd
                new_sum = alt_sum - nums[index]
            
            backtrack(index + 1, count + 1, new_sum, product * nums[index])
        
        backtrack(0, 0, 0, 1)
        return max_product