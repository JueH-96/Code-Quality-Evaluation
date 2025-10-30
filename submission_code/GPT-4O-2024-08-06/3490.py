class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Initialize the maximum length of valid subsequence
        max_length = 1
        # Initialize the current length of valid subsequence
        current_length = 1
        
        # Iterate through the array starting from the second element
        for i in range(1, len(nums)):
            # Check if the sum of the current and previous element has the same parity
            # as the sum of the previous and the element before that
            if (nums[i] + nums[i - 1]) % 2 == (nums[i - 1] + nums[i - 2]) % 2:
                # If it does, increase the current valid subsequence length
                current_length += 1
            else:
                # Otherwise, reset the current valid subsequence length
                current_length = 2
            
            # Update the maximum length found so far
            max_length = max(max_length, current_length)
        
        return max_length