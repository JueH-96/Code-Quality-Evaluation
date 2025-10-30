from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Helper function to compute count for a given index with left distance L and right distance R.
        # It returns: sum_{a=1}^{min(L,k)} min(R, k+1 - a)
        def count_subarrays(L: int, R: int, k: int) -> int:
            m = L if L < k else k  # m = min(L, k)
            # t is the threshold: for a <= t, we can take full R; for a > t, term = (k+1 - a)
            t = k + 1 - R
            if t <= 0:
                # For all a = 1..m, (k+1 - a) < R, so sum is sum_{a=1}^m (k+1-a).
                return m * (k + 1) - (m * (m + 1)) // 2
            elif t >= m:
                # For all a = 1..m, we have a <= m <= t, so k+1 - a >= R, hence each term is R.
                return m * R
            else:
                # For a = 1 to t, term is R; for a = t+1 to m, term is (k+1 - a)
                part1 = t * R
                cnt = m - t  # number of terms where a > t
                # Sum_{a=t+1}^{m} (k+1 - a)
                # Use arithmetic sum: = cnt*(k+1) - sum_{a=t+1}^{m} a
                # And sum_{a=t+1}^{m} a = (m*(m+1) - t*(t+1)) // 2
                part2 = cnt * (k + 1) - ((m * (m + 1) - t * (t + 1)) // 2)
                return part1 + part2
        
        # Compute distances for minimum contributions:
        left_min = [0] * n  # number of choices for left boundary for which nums[i] remains the minimum.
        right_min = [0] * n  # number of choices for right boundary.
        stack = []
        # For left_min, we want previous less element (strictly less)
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            if stack:
                left_min[i] = i - stack[-1]
            else:
                left_min[i] = i + 1
            stack.append(i)
            
        stack = []
        # For right_min, use next less or equal element.
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] >= nums[i]:
                stack.pop()
            if stack:
                right_min[i] = stack[-1] - i
            else:
                right_min[i] = n - i
            stack.append(i)
        
        # Compute distances for maximum contributions:
        left_max = [0] * n
        right_max = [0] * n
        stack = []
        # For left_max, use previous greater element (strictly greater)
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            if stack:
                left_max[i] = i - stack[-1]
            else:
                left_max[i] = i + 1
            stack.append(i)
        
        stack = []
        # For right_max, use next greater or equal element.
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                right_max[i] = stack[-1] - i
            else:
                right_max[i] = n - i
            stack.append(i)
        
        total_sum = 0
        # Sum contributions from minimums
        for i in range(n):
            cnt = count_subarrays(left_min[i], right_min[i], k)
            total_sum += nums[i] * cnt
            
        # Sum contributions from maximums
        for i in range(n):
            cnt = count_subarrays(left_max[i], right_max[i], k)
            total_sum += nums[i] * cnt
            
        return total_sum