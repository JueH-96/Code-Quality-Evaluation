class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_val = max(nums)
        max_positions = [i for i, num in enumerate(nums) if num == max_val]
        n = len(max_positions)
        if n < k:
            return 0
        
        result = 0
        count = 0
        left = 0
        
        for right in range(len(nums)):
            if nums[right] == max_val:
                count += 1
            
            while count >= k:
                result += len(nums) - right
                if nums[left] == max_val:
                    count -= 1
                left += 1
        
        return result