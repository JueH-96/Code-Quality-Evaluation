class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        from sortedcontainers import SortedList
        left = 0
        max_val = nums[0]
        min_val = nums[0]
        result = 0
        sl = SortedList([nums[0]])
        for right in range(1, len(nums)):
            sl.add(nums[right])
            max_val = max(max_val, nums[right])
            min_val = min(min_val, nums[right])
            while max_val - min_val > 2:
                sl.remove(nums[left])
                left += 1
                max_val = sl[-1]
                min_val = sl[0]
            result += right - left + 1
        return result