class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_sum = 0
        
        for i in range(n):
            is_good = True
            
            # Check if i-k exists and if nums[i] > nums[i-k]
            if i - k >= 0:
                if nums[i] <= nums[i - k]:
                    is_good = False
            
            # Check if i+k exists and if nums[i] > nums[i+k]
            if i + k < n:
                if nums[i] <= nums[i + k]:
                    is_good = False
            
            if is_good:
                total_sum += nums[i]
        
        return total_sum