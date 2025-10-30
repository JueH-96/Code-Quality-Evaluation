class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        max_num = max(nums)
        count = 0
        left = 0
        freq = 0
        
        for right in range(len(nums)):
            if nums[right] == max_num:
                freq += 1
            
            while freq >= k:
                count += len(nums) - right
                if nums[left] == max_num:
                    freq -= 1
                left += 1
        
        return count