class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_val = max(nums)
        n = len(nums)
        total_subarrays = 0
        for i in range(n):
            max_val_count = 0
            for j in range(i, n):
                if nums[j] == max_val:
                    max_val_count += 1
                if max_val_count >= k:
                    total_subarrays += (n - j)
                    break
        return total_subarrays