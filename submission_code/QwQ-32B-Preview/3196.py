from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        def can_achieve(f):
            if f == 0:
                return True
            for l in range(n - f + 1):
                r = l + f
                if f % 2 == 1:
                    # For odd f
                    sum_abs = prefix[r] - prefix[l + f // 2 + 1] - (prefix[l + f // 2] - prefix[l])
                else:
                    # For even f
                    sum_abs = prefix[r] + prefix[l] - 2 * prefix[l + f // 2]
                if sum_abs <= k:
                    return True
            return False
        
        # Binary search for the maximum achievable frequency
        low, high = 1, n
        best = 0
        while low <= high:
            mid = (low + high) // 2
            if can_achieve(mid):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best