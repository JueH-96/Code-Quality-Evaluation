class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        
        # Apply all queries using difference array
        for l, r in queries:
            diff[l] += 1
            diff[r + 1] -= 1
        
        # Check if each position can be decremented enough times
        count = 0
        for i in range(n):
            count += diff[i]
            if count < nums[i]:
                return False
        
        return True