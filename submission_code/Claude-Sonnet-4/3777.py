class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        def backtrack(index, current_sum, current_product, subsequence_length):
            # Base case: we've considered all elements
            if index == len(nums):
                # Check if we have a non-empty subsequence with the target sum
                if subsequence_length > 0 and current_sum == k and current_product <= limit:
                    return current_product
                return -1
            
            # Option 1: Don't include current element
            result = backtrack(index + 1, current_sum, current_product, subsequence_length)
            
            # Option 2: Include current element
            if nums[index] == 0:
                # Special case for 0: including it makes product 0
                new_product = 0
            else:
                new_product = current_product * nums[index]
            
            # Only proceed if product doesn't exceed limit
            if new_product <= limit:
                # Calculate new sum based on position in subsequence (0-indexed)
                if subsequence_length % 2 == 0:  # Even position in subsequence
                    new_sum = current_sum + nums[index]
                else:  # Odd position in subsequence
                    new_sum = current_sum - nums[index]
                
                option2_result = backtrack(index + 1, new_sum, new_product, subsequence_length + 1)
                result = max(result, option2_result)
            
            return result
        
        return backtrack(0, 0, 1, 0)