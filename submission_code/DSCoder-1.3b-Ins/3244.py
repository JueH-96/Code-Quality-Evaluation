class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        i = 0
        while i < len(nums) - 1:
            if nums[i] > 0 and nums[i + 1] > 0:
                nums.insert(len(nums), nums[i] % nums[i + 1])
                del nums[i]
                del nums[i]
            else:
                i += 1
        if len(nums) > 0 and nums[-1] > 0:
            nums.append(nums[-1] % nums[0])
        return len(nums)