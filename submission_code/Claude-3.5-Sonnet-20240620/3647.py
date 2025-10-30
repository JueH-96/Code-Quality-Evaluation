class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        def can_zero(k):
            diff = [0] * (n + 1)
            for i in range(k):
                l, r = queries[i]
                diff[l] += 1
                diff[r + 1] -= 1
            
            curr = 0
            for i in range(n):
                curr += diff[i]
                if curr < nums[i]:
                    return False
            return True
        
        left, right = 0, len(queries)
        while left < right:
            mid = (left + right + 1) // 2
            if can_zero(mid):
                left = mid
            else:
                right = mid - 1
        
        return left if left > 0 else -1