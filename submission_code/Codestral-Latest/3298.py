class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        # Sort the array to make it easier to find consecutive elements
        nums.sort()

        # Initialize variables to keep track of the longest consecutive sequence
        max_length = 1
        current_length = 1

        # Iterate through the sorted array
        for i in range(1, nums.length):
            # If the current element is one more than the previous element, it's part of the sequence
            if nums[i] == nums[i - 1] + 1:
                current_length += 1
            else:
                # Update the maximum length if the current sequence is longer
                max_length = max(max_length, current_length)
                # Reset the current length
                current_length = 1

        # Update the maximum length one last time in case the longest sequence is at the end
        max_length = max(max_length, current_length)

        return max_length