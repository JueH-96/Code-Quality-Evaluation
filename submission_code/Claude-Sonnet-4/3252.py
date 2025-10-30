class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Try all possible subarrays to remove
        for i in range(n):  # start index of subarray to remove
            for j in range(i, n):  # end index of subarray to remove
                # Create the remaining array after removing subarray [i:j+1]
                remaining = nums[:i] + nums[j+1:]
                
                # Check if remaining array is strictly increasing
                if self.isStrictlyIncreasing(remaining):
                    count += 1
        
        return count
    
    def isStrictlyIncreasing(self, arr):
        if len(arr) <= 1:
            return True
        
        for i in range(1, len(arr)):
            if arr[i] <= arr[i-1]:
                return False
        
        return True