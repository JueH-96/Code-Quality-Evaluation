class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        count = 0
        
        # Check each possible group size from 0 to n
        for k in range(n + 1):
            valid = True
            
            # For selecting first k students:
            # Check if the largest selected student satisfies nums[i] < k
            if k > 0 and nums[k-1] >= k:
                valid = False
            
            # Check if the smallest non-selected student satisfies nums[i] > k  
            if k < n and nums[k] <= k:
                valid = False
                
            if valid:
                count += 1
                
        return count