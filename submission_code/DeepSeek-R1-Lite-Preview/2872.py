class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # Step 1: Group the array into non-decreasing subarrays and sum each group
        group_sums = []
        current_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i] >= nums[i - 1]:
                current_sum += nums[i]
            else:
                group_sums.append(current_sum)
                current_sum = nums[i]
        group_sums.append(current_sum)
        
        # Step 2: Use a stack to merge group sums where previous <= current
        stack = []
        for sum_val in group_sums:
            stack.append(sum_val)
            while len(stack) >= 2 and stack[-2] <= stack[-1]:
                merged_sum = stack.pop() + stack.pop()
                stack.append(merged_sum)
        
        # Step 3: Find the maximum sum in the stack
        return max(stack)