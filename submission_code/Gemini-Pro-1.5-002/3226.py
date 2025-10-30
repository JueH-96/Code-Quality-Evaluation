class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()
        arr = []
        left = 0
        right = len(nums) - 1
        while left < right:
            arr.append(nums[right])
            right -= 1
            arr.append(nums[left])
            left += 1
        return arr