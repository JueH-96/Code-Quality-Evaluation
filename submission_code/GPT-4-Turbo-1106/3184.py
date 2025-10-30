class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        # Initialize the maximum sum to the first element, as a subsequence of length 1 is always balanced
        max_sum = nums[0]
        
        # Create a stack to keep track of the potential elements of the balanced subsequence
        stack = []
        
        for i, num in enumerate(nums):
            # If the stack is empty or the current number satisfies the balanced condition with the top of the stack
            while stack and num - nums[stack[-1]] >= i - stack[-1]:
                # Pop the top of the stack
                stack.pop()
            # If the stack is not empty, add the current number to the sum
            if stack:
                max_sum = max(max_sum, nums[stack[-1]] + num)
            # Push the current index onto the stack
            stack.append(i)
        
        return max_sum