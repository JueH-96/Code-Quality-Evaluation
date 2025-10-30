class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums)):
            max_val = max(nums[i:])
            if nums[i] == max_val:
                left = i
                right = i
                while left >= 0 and nums[left] == max_val:
                    left -= 1
                while right < len(nums) and nums[right] == max_val:
                    right += 1
                count += (right - i) * (i - left)
        return count