class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        left = 0
        current_max = nums[0]
        current_min = nums[0]
        result = 0
        for right in range(n):
            if right > 0:
                if nums[right] > current_max:
                    current_max = nums[right]
                if nums[right] < current_min:
                    current_min = nums[right]
            while current_max - current_min > 2:
                if nums[left] == current_max:
                    current_max = max(nums[left+1:right+1])
                if nums[left] == current_min:
                    current_min = min(nums[left+1:right+1])
                left += 1
            result += (right - left + 1)
        return result