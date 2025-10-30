class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        dominant = max(set(nums), key=nums.count)
        dominant_count = nums.count(dominant)
        left_count = 0
        for i in range(len(nums) - 1):
            if nums[i] == dominant:
                left_count += 1
            if left_count * 2 > i + 1 and (dominant_count - left_count) * 2 > len(nums) - i - 1:
                return i
        return -1