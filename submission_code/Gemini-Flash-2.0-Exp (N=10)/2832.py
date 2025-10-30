class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_len = 0
        for target_val in set(nums):
            left = 0
            cost = 0
            for right in range(n):
                if nums[right] != target_val:
                    cost += 1
                while cost > k:
                    if nums[left] != target_val:
                        cost -= 1
                    left += 1
                max_len = max(max_len, right - left + 1)
        return max_len