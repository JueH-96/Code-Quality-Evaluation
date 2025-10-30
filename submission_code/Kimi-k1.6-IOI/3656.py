class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        min_ops = float('inf')
        for i in range(n + 1):
            sub = nums[i:]
            if len(set(sub)) == len(sub):
                ops = (i + 2) // 3
                if ops < min_ops:
                    min_ops = ops
        return min_ops