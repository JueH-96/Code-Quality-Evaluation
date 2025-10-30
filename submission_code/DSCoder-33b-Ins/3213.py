class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_num = max(nums)
        count = [0] * (max_num + 1)
        left = 0
        right = 0
        result = 0
        while right < len(nums):
            count[nums[right]] += 1
            while count[max_num] < k:
                count[nums[left]] -= 1
                left += 1
            result += left
            right += 1
        return result