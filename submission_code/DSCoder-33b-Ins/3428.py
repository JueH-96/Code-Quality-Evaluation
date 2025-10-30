class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        nums.sort()
        xor = 0
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                xor ^= nums[i]
        return xor