from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * n
        prefix_sum[0] = nums[0]
        for i in range(1, n):
            prefix_sum[i] = prefix_sum[i-1] + nums[i]
            
        def is_possible(freq):
            for i in range(n):
                if i - freq + 1 < 0:
                    continue
                cost = freq * nums[i] - (prefix_sum[i] - (prefix_sum[i - freq] if i - freq >= 0 else 0))
                if cost <= k:
                    return True
            return False
            
        low = 0
        high = n
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            if mid == 0:
                ans = max(ans, mid)
                low = mid + 1
                continue
            if is_possible(mid):
                ans = max(ans, mid)
                low = mid + 1
            else:
                high = mid - 1
        return ans