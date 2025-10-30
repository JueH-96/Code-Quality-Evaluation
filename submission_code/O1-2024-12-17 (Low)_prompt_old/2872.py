class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # We'll process the array from right-to-left, merging as long as
        # the left element is <= the current sum (representing the "right" part).
        # This is because the operation allows us to merge nums[i] into nums[i+1]
        # if nums[i] <= nums[i+1]. In a right-to-left sweep, the "current sum" acts
        # like nums[i+1], and we check whether nums[i] is <= that sum.
        
        current_sum = 0
        max_value = 0
        for i in range(len(nums) - 1, -1, -1):
            # If the current element can be merged into the running sum, do so;
            # otherwise start a new sum from the current element.
            if nums[i] <= current_sum:
                current_sum += nums[i]
            else:
                current_sum = nums[i]
            max_value = max(max_value, current_sum)
        
        return max_value