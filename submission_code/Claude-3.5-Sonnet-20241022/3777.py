class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        n = len(nums)
        max_product = -1
        
        def backtrack(index, curr_seq, curr_alt_sum, curr_product):
            nonlocal max_product
            
            # If we have a valid subsequence
            if len(curr_seq) > 0:
                # Check if alternating sum equals k and product is within limit
                if curr_alt_sum == k and curr_product <= limit:
                    max_product = max(max_product, curr_product)
            
            # Base case: if we've processed all numbers
            if index >= n:
                return
            
            # For each position, we have two choices:
            # 1. Include the current number
            curr_num = nums[index]
            # Calculate new alternating sum based on position in subsequence
            new_alt_sum = curr_alt_sum
            if len(curr_seq) % 2 == 0:
                new_alt_sum += curr_num
            else:
                new_alt_sum -= curr_num
            
            # Only proceed if product doesn't exceed limit
            new_product = curr_product * curr_num
            if new_product <= limit:
                curr_seq.append(curr_num)
                backtrack(index + 1, curr_seq, new_alt_sum, new_product)
                curr_seq.pop()
            
            # 2. Skip the current number
            backtrack(index + 1, curr_seq, curr_alt_sum, curr_product)
        
        # Start backtracking
        backtrack(0, [], 0, 1)
        
        return max_product