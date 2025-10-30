class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Compute the difference array
        d = [target[i] - nums[i] for i in range(n)]
        
        res = abs(d[0])
        
        for i in range(1, n):
            if d[i] > d[i-1]:
                res += d[i] - d[i-1]
        
        return res