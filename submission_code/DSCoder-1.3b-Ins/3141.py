class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        left, right = 0, 0
        current_sum = nums[0]
        min_length = float('inf')
        while right < len(nums):
            if current_sum < target:
                right += 1
                if right < len(nums):
                    current_sum += nums[right]
            else:
                min_length = min(min_length, right - left + 1)
                current_sum -= nums[left]
                left += 1
        return min_length if min_length != float('inf') else -1