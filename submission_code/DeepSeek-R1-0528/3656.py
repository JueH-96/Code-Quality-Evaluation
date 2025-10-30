class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        max_ops = (n + 2) // 3
        for k in range(max_ops + 1):
            start_index = 3 * k
            sub = nums[start_index:]
            if len(sub) == len(set(sub)):
                return k
        return max_ops