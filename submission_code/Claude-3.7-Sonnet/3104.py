class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()  # Sort to easily check conditions
        ways = 0
        
        # Check if selecting 0 students makes everyone happy
        if nums[0] > 0:
            ways += 1
        
        # Check for selecting 1 to n-1 students
        for k in range(1, n):
            if nums[k-1] < k and (k == n - 1 or nums[k] > k):
                ways += 1
        
        # Check if selecting all n students makes everyone happy
        if nums[n-1] < n:
            ways += 1
        
        return ways