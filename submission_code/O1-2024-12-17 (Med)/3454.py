class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        
        # Calculate the delta array: delta[i] = target[i] - nums[i]
        delta = [target[i] - nums[i] for i in range(n)]
        
        # We'll compute the "extended" absolute-difference sum:
        # |delta[0] - 0| + sum(|delta[i] - delta[i-1]| for i=1..n-1) + |0 - delta[n-1]|
        # The answer is half of this sum.
        
        total_diff = abs(delta[0])  # |delta[0] - 0|
        for i in range(1, n):
            total_diff += abs(delta[i] - delta[i-1])
        total_diff += abs(delta[n-1])  # |delta[n-1] - 0|
        
        return total_diff // 2