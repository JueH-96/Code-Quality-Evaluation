class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        left = 0
        count = 0
        max_val = min_val = nums[0]

        for right in range(n):
            max_val = max(max_val, nums[right])
            min_val = min(min_val, nums[right])

            while max_val - min_val > 2:
                if nums[left] == max_val:
                    max_val = max(nums[left+1:right+1])
                if nums[left] == min_val:
                    min_val = min(nums[left+1:right+1])
                left += 1

            count += right - left + 1

        return count