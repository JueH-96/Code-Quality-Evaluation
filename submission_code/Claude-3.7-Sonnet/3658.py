class Solution:
    def minDifference(self, nums: List[int]) -> int:
        # If all elements are missing, we can make all elements the same
        if all(num == -1 for num in nums):
            return 0

        n = len(nums)
        
        # Compute the range of non-missing elements
        non_missing = [num for num in nums if num != -1]
        min_val, max_val = min(non_missing), max(non_missing)
        
        # Precompute the maximum difference between consecutive non-missing elements
        max_diff = 0
        for i in range(1, n):
            if nums[i] != -1 and nums[i-1] != -1:
                max_diff = max(max_diff, abs(nums[i] - nums[i-1]))
        
        # Function to check if a given maximum difference d is feasible
        def is_feasible(d):
            # Compute constraints for x and y
            x_min, x_max = 1, 10**9
            y_min, y_max = 1, 10**9
            
            for i in range(n):
                if nums[i] != -1:
                    # If this non-missing element has a missing neighbor,
                    # both x and y need to be within d of this element
                    if (i > 0 and nums[i-1] == -1) or (i < n-1 and nums[i+1] == -1):
                        x_min = max(x_min, nums[i] - d)
                        x_max = min(x_max, nums[i] + d)
                        y_min = max(y_min, nums[i] - d)
                        y_max = min(y_max, nums[i] + d)
            
            # Check if the constraints for x and y are satisfiable
            if x_min > x_max or y_min > y_max:
                return False
            
            # Check if |x - y| can be at most d
            if x_min > y_max + d or y_min > x_max + d:
                return False
            
            # Check if all differences between consecutive non-missing elements are at most d
            for i in range(1, n):
                if nums[i] != -1 and nums[i-1] != -1 and abs(nums[i] - nums[i-1]) > d:
                    return False
            
            return True
        
        # Binary search for the minimum feasible maximum difference
        left, right = 0, max(max_diff, max_val - min_val)
        while left < right:
            mid = (left + right) // 2
            if is_feasible(mid):
                right = mid
            else:
                left = mid + 1
        
        return left