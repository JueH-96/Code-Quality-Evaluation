class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Initialize the maximum length of valid subsequence
        max_length = 1
        current_length = 1
        
        # Iterate through the array to find the longest valid subsequence
        for i in range(1, nums.length):
            if (nums[i-1] + nums[i]) % 2 == (nums[i] + nums[i+1]) % 2:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 1
        
        return max_length