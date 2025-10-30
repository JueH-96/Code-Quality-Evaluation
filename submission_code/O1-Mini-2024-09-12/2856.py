class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        from collections import defaultdict

        total_unique = len(set(nums))
        count = defaultdict(int)
        res = 0
        left = 0
        unique_in_window = 0

        for right, num in enumerate(nums):
            count[num] += 1
            if count[num] == 1:
                unique_in_window += 1

            while unique_in_window == total_unique:
                res += len(nums) - right
                count[nums[left]] -= 1
                if count[nums[left]] == 0:
                    unique_in_window -= 1
                left += 1

        return res