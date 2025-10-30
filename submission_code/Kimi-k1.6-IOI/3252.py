class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Compute left array where left[i] is True if nums[0..i] is strictly increasing
        left = [False] * n
        left[0] = True
        for i in range(1, n):
            left[i] = left[i-1] and (nums[i] > nums[i-1])
        
        # Compute right array where right[i] is True if nums[i..n-1] is strictly increasing
        right = [False] * n
        right[-1] = True
        for i in range(n-2, -1, -1):
            right[i] = right[i+1] and (nums[i] < nums[i+1])
        
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                # Check left part (0 to i-1)
                left_valid = True
                if i > 0:
                    left_valid = left[i-1]
                
                # Check right part (j+1 to n-1)
                right_valid = True
                if j < n - 1:
                    right_valid = right[j+1]
                
                # Check cross condition if both parts exist
                cross_valid = True
                if i > 0 and j < n - 1:
                    cross_valid = (nums[i-1] < nums[j+1])
                
                if left_valid and right_valid and cross_valid:
                    count += 1
        
        return count