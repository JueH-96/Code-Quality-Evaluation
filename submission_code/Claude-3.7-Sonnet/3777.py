class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        n = len(nums)
        max_product = -1
        
        def backtrack(idx, alt_sum, product, length):
            nonlocal max_product
            
            # Check if current subsequence satisfies the conditions
            if length > 0 and alt_sum == k and product <= limit:
                max_product = max(max_product, product)
            
            # If we've considered all elements, return
            if idx == n:
                return
            
            # Skip the current element
            backtrack(idx + 1, alt_sum, product, length)
            
            # Take the current element
            new_product = product * nums[idx]
            if new_product <= limit:
                if length % 2 == 0:  # Even position (add)
                    backtrack(idx + 1, alt_sum + nums[idx], new_product, length + 1)
                else:  # Odd position (subtract)
                    backtrack(idx + 1, alt_sum - nums[idx], new_product, length + 1)
        
        backtrack(0, 0, 1, 0)
        return max_product