class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_val = max(nums)
        n = len(nums)
        total = n * (n + 1) // 2
        
        less_than_k = 0
        left = 0
        count = 0
        
        for right in range(n):
            if nums[right] == max_val:
                count += 1
            # Ensure the window [left, right] has less than k max elements
            while count >= k:
                if nums[left] == max_val:
                    count -= 1
                left += 1
            # Add the number of valid subarrays ending at right
            less_than_k += (right - left + 1)
        
        return total - less_than_k