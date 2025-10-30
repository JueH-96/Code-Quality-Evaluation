class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0

        # Initialize the longest valid subsequence length
        max_length = 1

        # Initialize the current length of the valid subsequence
        current_length = 1

        # Iterate through the array starting from the second element
        for i in range(1, n):
            # Check if the current element and the previous element satisfy the condition
            if (nums[i] + nums[i - 1]) % 2 == 0:
                current_length += 1
            else:
                # Update the maximum length if the current length is greater
                max_length = max(max_length, current_length)
                current_length = 1

        # Update the maximum length for the last subsequence
        max_length = max(max_length, current_length)

        return max_length