class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        total = 0
        n = len(nums)
        for i in range(n):
            # Assume element is good until disproved
            is_good = True
            
            # Check left neighbor at i-k if exists
            if i - k >= 0:
                if not (nums[i] > nums[i-k]):
                    is_good = False
            
            # Check right neighbor at i+k if exists
            if i + k < n:
                if not (nums[i] > nums[i+k]):
                    is_good = False
            
            # Add to total if still good
            if is_good:
                total += nums[i]
                
        return total