from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        # prefix sums: ps[i] = sum of nums[0..i-1]
        ps = [0] * (n + 1)
        for i in range(n):
            ps[i+1] = ps[i] + nums[i]
        
        ans = 1
        # For each right endpoint r, binary-search the smallest l in [0..r]
        # such that the cost to make nums[l..r] all equal to their median <= k.
        for r in range(n):
            lo, hi = 0, r
            best = r  # at worst we take only nums[r]
            while lo <= hi:
                mid = (lo + hi) // 2
                # median index of the window [mid..r]
                m = (mid + r) // 2
                # cost to bring nums[mid..r] to nums[m]
                # left part [mid..m]
                left_count = m - mid + 1
                left_sum = ps[m+1] - ps[mid]
                left_cost = nums[m] * left_count - left_sum
                # right part [m+1..r]
                right_count = r - m
                right_sum = ps[r+1] - ps[m+1]
                right_cost = right_sum - nums[m] * right_count
                total_cost = left_cost + right_cost
                
                if total_cost <= k:
                    best = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            
            # window size = r - best + 1
            ans = max(ans, r - best + 1)
        
        return ans