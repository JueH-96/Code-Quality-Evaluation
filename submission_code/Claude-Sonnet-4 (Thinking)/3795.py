class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        def canMakeZero(k):
            arr = nums[:]
            for i in range(k):
                l, r, val = queries[i]
                for j in range(l, r + 1):
                    if arr[j] >= val:
                        arr[j] -= val
            return all(x == 0 for x in arr)
        
        # Binary search for minimum k
        left, right = 0, len(queries)
        answer = -1
        
        while left <= right:
            mid = (left + right) // 2
            if canMakeZero(mid):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return answer