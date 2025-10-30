class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_length = 0
        left = 0
        changes = 0
        
        for right in range(1, n):
            if nums[right] != nums[right - 1]:
                changes += 1
            
            while changes > k:
                if nums[left] != nums[left + 1]:
                    changes -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length