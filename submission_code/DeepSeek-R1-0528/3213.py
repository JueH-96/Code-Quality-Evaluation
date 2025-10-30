class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_val = max(nums)
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        left = 0
        count_max = 0
        less_than_k = 0
        
        for right in range(n):
            if nums[right] == max_val:
                count_max += 1
            while left <= right and count_max >= k:
                if nums[left] == max_val:
                    count_max -= 1
                left += 1
            less_than_k += (right - left + 1)
        
        return total_subarrays - less_than_k