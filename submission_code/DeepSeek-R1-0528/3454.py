class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        d = [target[i] - nums[i] for i in range(n)]
        total_ops = abs(d[0]) + abs(d[-1])
        for i in range(1, n):
            total_ops += abs(d[i] - d[i-1])
        return total_ops // 2