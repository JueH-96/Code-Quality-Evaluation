class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Binary search on the answer
        def can_achieve_max_diff(max_diff):
            # Try to fill -1s such that no adjacent difference exceeds max_diff
            arr = nums[:]
            
            for i in range(n):
                if arr[i] == -1:
                    # Find valid range for this position
                    min_val, max_val = 1, 10**9
                    
                    # Constraint from left neighbor
                    if i > 0 and arr[i-1] != -1:
                        min_val = max(min_val, arr[i-1] - max_diff)
                        max_val = min(max_val, arr[i-1] + max_diff)
                    
                    # Constraint from right neighbor
                    if i < n-1 and arr[i+1] != -1:
                        min_val = max(min_val, arr[i+1] - max_diff)
                        max_val = min(max_val, arr[i+1] + max_diff)
                    
                    if min_val > max_val:
                        return False
                    
                    # Choose a value in the valid range
                    arr[i] = min_val
            
            # Check if all differences are within max_diff
            for i in range(n-1):
                if abs(arr[i] - arr[i+1]) > max_diff:
                    return False
            
            return True
        
        left, right = 0, 10**9
        while left < right:
            mid = (left + right) // 2
            if can_achieve_max_diff(mid):
                right = mid
            else:
                left = mid + 1
        
        return left