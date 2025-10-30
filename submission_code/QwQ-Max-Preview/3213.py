class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_num = max(nums)
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        left = 0
        current_count = 0
        result_less = 0
        
        for right in range(n):
            if nums[right] == max_num:
                current_count += 1
            
            # Move left pointer to ensure current_count < k
            while current_count >= k:
                if nums[left] == max_num:
                    current_count -= 1
                left += 1
            
            # Add the number of valid subarrays ending at 'right'
            result_less += (right - left + 1)
        
        return total_subarrays - result_less