class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        nums.sort()
        left, right = 0, len(nums) - 1
        while left < right:
            if nums[left] > 0 and nums[right] > 0:
                nums.append(nums[right] % nums[left])
                nums.pop(right)
                nums.pop(left)
                nums.sort()
                left, right = 0, len(nums) - 1
            elif nums[left] <= 0:
                left += 1
            elif nums[right] <= 0:
                right -= 1
        return len(nums)