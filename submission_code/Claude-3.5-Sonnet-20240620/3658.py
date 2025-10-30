class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        non_missing = [num for num in nums if num != -1]
        
        if not non_missing:
            return 0
        
        non_missing.sort()
        
        def check(diff):
            prev = non_missing[0]
            count = 1
            for num in non_missing[1:]:
                if num - prev > diff:
                    count += 1
                    prev = num
            return count <= n - nums.count(-1) + 1
        
        left, right = 0, 10**9
        while left < right:
            mid = (left + right) // 2
            if check(mid):
                right = mid
            else:
                left = mid + 1
        
        return left