class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_num = max(nums)
        count = 0
        res = 0
        left = 0
        for right, num in enumerate(nums):
            if num == max_num:
                count += 1
            while count >= k:
                res += len(nums) - right
                if nums[left] == max_num:
                    count -= 1
                left += 1
        return res