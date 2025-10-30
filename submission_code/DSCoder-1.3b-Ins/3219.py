class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        nums.sort()
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if abs(nums[i] - nums[j]) > limit:
                    nums[i], nums[j] = nums[j], nums[i]
                    break
        return nums