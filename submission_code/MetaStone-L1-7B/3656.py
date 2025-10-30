class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        max_k = (len(nums) + 2) // 3
        for k in range(0, max_k + 1):
            sub = nums[3*k:]
            if len(sub) == 0:
                return k
            if len(sub) == len(set(sub)):
                return k
        return max_k