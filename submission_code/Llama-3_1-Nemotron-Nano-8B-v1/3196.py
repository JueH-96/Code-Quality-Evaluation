from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        
        left, right = 1, n
        ans = 0
        
        while left <= right:
            mid = (left + right) // 2
            found = False
            # Check all possible windows of size mid
            for i in range(n - mid + 1):
                l = i
                r = i + mid - 1
                m = (l + r) // 2
                # Calculate the cost to make all elements in [l, r] equal to nums[m]
                left_part = nums[m] * (m - l + 1) - (pre[m + 1] - pre[l])
                right_part = (pre[r + 1] - pre[m + 1]) - nums[m] * (r - m)
                total_cost = left_part + right_part
                if total_cost <= k:
                    found = True
                    break
            if found:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans