class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Initialize the maximum score and the count of subarrays
        max_score, count = 0, 0
        # Iterate over the array
        for num in nums:
            # If the current number is greater than the maximum score, update the maximum score and reset the count
            if num > max_score:
                max_score = num
                count = 1
            # If the current number is equal to the maximum score, increment the count
            elif num == max_score:
                count += 1
        # Return the count of subarrays
        return count