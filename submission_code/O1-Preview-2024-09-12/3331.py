class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        index = 0
        while index < len(nums) and nums[index] < k:
            index += 1
        return index