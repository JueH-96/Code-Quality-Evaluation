class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        
        # Check if it's possible with all queries
        def canMakeZero(k):
            # For each position, calculate the maximum we can decrement
            decrements = [0] * n
            
            for i in range(k):
                l, r, val = queries[i]
                for j in range(l, r + 1):
                    decrements[j] += val
            
            # Check if we can make all elements zero
            for i in range(n):
                if nums[i] > decrements[i]:
                    return False
            return True
        
        # Binary search on the number of queries
        left, right = 0, len(queries)
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if canMakeZero(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result