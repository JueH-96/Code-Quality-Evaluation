class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        diff = [t - n for n, t in zip(nums, target)]
        prefix_sum = [0]
        for d in diff:
            prefix_sum.append(prefix_sum[-1] + d)
        
        stack = []
        operations = 0
        
        for i, curr_sum in enumerate(prefix_sum):
            while stack and curr_sum < prefix_sum[stack[-1]]:
                operations = max(operations, prefix_sum[stack.pop()] - curr_sum)
            stack.append(i)
        
        return operations