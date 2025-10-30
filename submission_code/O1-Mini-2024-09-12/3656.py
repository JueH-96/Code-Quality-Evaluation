class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        for k in range(0, (len(nums) + 2) // 3 + 1):
            if k * 3 >= len(nums):
                return k
            remaining = nums[k * 3:]
            if len(set(remaining)) == len(remaining):
                return k
        return (len(nums) + 2) // 3 + 1