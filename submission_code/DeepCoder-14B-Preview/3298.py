class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        if not nums:
            return 0
        nums.sort()
        n = len(nums)
        dp0 = [1] * n
        dp1 = [1] * n
        
        for i in range(1, n):
            # Update dp0[i]
            if nums[i] == nums[i-1] + 1:
                dp0[i] = dp0[i-1] + 1
            
            # Update dp1[i]
            if nums[i] == nums[i-1]:
                dp1[i] = max(dp1[i], dp0[i-1] + 1)
            if nums[i] == nums[i-1] + 1:
                dp1[i] = max(dp1[i], dp1[i-1] + 1)
        
        max_len = max(max(dp0), max(dp1))
        return max_len