class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        def is_strictly_increasing(arr):
            if not arr:
                return True
            for i in range(1, len(arr)):
                if arr[i] <= arr[i-1]:
                    return False
            return True
        
        n = len(nums)
        count = 0
        
        # Try all possible subarrays
        for i in range(n):
            for j in range(i, n):
                # Create array without the subarray from i to j
                remaining = nums[:i] + nums[j+1:]
                if is_strictly_increasing(remaining):
                    count += 1
                    
        return count