class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        while len(nums) > 1:
            i, j = 0, 1
            while i < len(nums) and j < len(nums):
                if nums[i] > 0 and nums[j] > 0:
                    nums.append(nums[i] % nums[j])
                    nums.pop(j)
                    nums.pop(i)
                    break
                j += 1
                if j == len(nums):
                    i += 1
                    j = i + 1
        return len(nums)