class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        
        count = 0
        
        # Case 1: Select 0 students
        # All students must have nums[i] > 0
        if nums[0] > 0:
            count += 1
        
        # Case 2: Select k students (1 <= k <= n-1)
        for k in range(1, n):
            # We select the first k students (indices 0 to k-1)
            # The k-th student (index k-1) must have nums[k-1] < k
            # The (k+1)-th student (index k) must have nums[k] > k
            if nums[k-1] < k and nums[k] > k:
                count += 1
        
        # Case 3: Select all n students
        # All students must have nums[i] < n
        if nums[n-1] < n:
            count += 1
        
        return count