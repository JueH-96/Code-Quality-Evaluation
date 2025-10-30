class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        diff = [0] * (n + 2)  # Using n+2 to handle r+1 up to n
        
        for l, r in queries:
            diff[l] += 1
            diff[r + 1] -= 1
        
        current = 0
        count = [0] * n
        
        for i in range(n):
            current += diff[i]
            count[i] = current
        
        for j in range(n):
            if nums[j] > 0 and count[j] < nums[j]:
                return False
        
        return True