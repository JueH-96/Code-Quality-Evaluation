class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_num = max(nums)
        n = len(nums)
        result = 0
        
        left = 0
        count = 0
        
        for right in range(n):
            if nums[right] == max_num:
                count += 1
            
            while count >= k:
                result += n - right
                if nums[left] == max_num:
                    count -= 1
                left += 1
                
        return result