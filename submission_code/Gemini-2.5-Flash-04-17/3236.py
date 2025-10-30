from typing import List

class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Step 1: Find the length of the longest sequential prefix
        n = len(nums)
        
        # The prefix always starts with nums[0]
        # Initialize length to 1 (for nums[0])
        prefix_len = 1
        
        # Iterate from the second element (index 1) up to the end of the array
        # We are looking for the point where the sequential property (nums[i] == nums[i-1] + 1) breaks
        for i in range(1, n):
            # Check if the current element continues the sequence from the previous one
            if nums[i] == nums[i-1] + 1:
                # If it does, extend the length of the sequential prefix
                prefix_len += 1
            else:
                # If the sequence is broken, we've found the end of the longest sequential prefix
                # The length is `prefix_len` at this point (it includes elements up to index i-1)
                break # Exit the loop as the prefix ends
                
        # Step 2: Calculate the sum of the longest sequential prefix
        # The longest sequential prefix is the slice nums[0 : prefix_len]
        # Python slice nums[:prefix_len] correctly includes elements from index 0 up to (but not including) prefix_len
        # Since prefix_len is the *count* of elements, this slice includes indices 0, 1, ..., prefix_len - 1, which is correct.
        prefix_sum = sum(nums[:prefix_len])
        
        # Step 3: Find the smallest integer >= prefix_sum that is not in nums
        # Convert the nums array to a set for efficient lookups (O(1) average time complexity)
        nums_set = set(nums)
        
        # Start checking integers from the calculated prefix_sum
        current_check = prefix_sum
        
        # Keep checking subsequent integers as long as they are found in the set of numbers from nums
        while current_check in nums_set:
            current_check += 1
            
        # The first integer not found in the set (starting from prefix_sum) is the smallest missing integer
        # that is greater than or equal to the prefix sum.
        return current_check