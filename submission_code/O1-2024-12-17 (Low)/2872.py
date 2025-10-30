class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # We'll process the array from right to left and maintain a running sum.
        # If the element to the left (nums[i]) is <= the current running sum,
        # it can be merged into a single element by summation. Otherwise, we reset the running sum.
        # The result is the maximum running sum encountered.

        max_val = 0
        curr_sum = 0

        # Traverse from right to left
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] <= curr_sum:
                # Merge current element into the running sum
                curr_sum += nums[i]
            else:
                # Reset the running sum with the current element
                curr_sum = nums[i]
            # Track the maximum sum obtained
            max_val = max(max_val, curr_sum)

        return max_val