class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_num = max(nums)
        count = 0
        result = 0
        left = 0
        
        for right in range(n):
            if nums[right] == max_num:
                count += 1
            
            while count >= k:
                result += n - right
                if nums[left] == max_num:
                    count -= 1
                left += 1
        
        return result