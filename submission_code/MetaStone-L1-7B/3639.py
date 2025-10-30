class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        diff = [0] * (n + 2)  # Difference array to handle range updates
        
        for query in queries:
            l, r = query
            diff[l] += 1
            diff[r + 1] -= 1
        
        current = 0
        count = [0] * n
        for i in range(n):
            current += diff[i]
            count[i] = current
        
        for num in nums:
            if num > count[i]:
                return False
        return True