class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        candidates = []
        
        # Find potential palindromic candidates
        for i in range(n):
            if str(nums[i]) == str(nums[i])[::-1]:
                candidates.append(nums[i])
                
            if i > 0 and nums[i] != nums[i-1] and str(nums[i]-1) == str(nums[i]-1)[::-1]:
                candidates.append(nums[i]-1)
                
            if i < n-1 and nums[i] != nums[i+1] and str(nums[i]+1) == str(nums[i]+1)[::-1]:
                candidates.append(nums[i]+1)
                
        # Calculate cost for each candidate and return the minimum
        min_cost = float('inf')
        for candidate in candidates:
            cost = sum(abs(num - candidate) for num in nums)
            min_cost = min(min_cost, cost)
            
        return min_cost