class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()
        left, right = 0, len(nums) - 1
        min_avg = float('inf')
        while left < right:
            current_avg = (nums[left] + nums[right]) / 2.0
            if current_avg < min_avg:
                min_avg = current_avg
            left += 1
            right -= 1
        return min_avg