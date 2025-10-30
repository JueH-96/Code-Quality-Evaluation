class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        
        def is_sorted(arr):
            for i in range(len(arr) - 1):
                if arr[i] > arr[i+1]:
                    return False
            return True
        
        if is_sorted(nums):
            return 0
        
        shifted_nums = list(nums) # Create a copy to avoid modifying the original nums in each iteration
        for shifts in range(1, n):
            last_element = shifted_nums[-1]
            shifted_nums = [last_element] + shifted_nums[:-1]
            if is_sorted(shifted_nums):
                return shifts
                
        return -1