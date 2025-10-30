class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        if len(nums) == 0 or len(queries) == 0:
            return False
        
        minvals = [nums[0]]
        maxvals = [nums[0]]
        
        for i in range(1, len(nums)):
            minvals.append(min(minvals[i-1], nums[i]))
            maxvals.append(max(maxvals[i-1], nums[i]))
        
        # Reverse the arrays to maintain their functionality when indexed from the end
        minvals = minvals[::-1]
        maxvals = maxvals[::-1]
        
        for l, r in queries:
            min_val = min(maxvals[len(nums)-1-l], minvals[len(nums)-1-r])
            if min_val > r-l+1 or max_val > r-l+1:
                return False
            max_val = max(maxvals[len(nums)-1-l], minvals[len(nums)-1-r])
            
            for i in range(l, r+1):
                nums[i] -= 1
            
        return all(num == 0 for num in nums)