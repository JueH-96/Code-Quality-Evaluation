class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
            
        # Find where array is not sorted (pivot point)
        pivot = -1
        for i in range(n-1):
            if nums[i] > nums[i+1]:
                if pivot != -1:  # More than one pivot means impossible
                    return -1
                pivot = i
                
        if pivot == -1:  # Already sorted
            return 0
            
        # Check if rotating at pivot point makes array sorted
        if nums[pivot+1:] + nums[:pivot+1] == sorted(nums):
            return n - (pivot + 1)
            
        return -1