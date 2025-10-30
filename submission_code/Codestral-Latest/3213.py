class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_val = max(nums)
        count = 0
        n = len(nums)

        for i in range(n):
            max_count = 0
            for j in range(i, n):
                if nums[j] == max_val:
                    max_count += 1
                if max_count >= k:
                    count += 1

        return count