import bisect
from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()

        # Calculate prefix sums
        # prefix_sum[i] stores the sum of the first i elements (nums[0]...nums[i-1])
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        # Function to calculate cost for a window nums[l...r] with target nums[mid_idx]
        # The target is the median of the elements in the window
        # We pass l, r directly
        def calculate_cost(l: int, r: int) -> int:
            # Window indices are l, l+1, ..., r
            # Window length f = r - l + 1
            # Median index in the sorted array nums is mid_idx = l + (r - l) // 2
            mid_idx = l + (r - l) // 2
            target_val = nums[mid_idx]

            # Cost to change elements from l to mid_idx - 1 to target_val
            # These elements are smaller than or equal to target_val
            # sum_{j=l}^{mid_idx-1} (target_val - nums[j])
            # = (mid_idx - l) * target_val - sum_{j=l}^{mid_idx-1} nums[j]
            # sum_{j=l}^{mid_idx-1} nums[j] = prefix_sum[mid_idx] - prefix_sum[l]
            left_cost = (mid_idx - l) * target_val - (prefix_sum[mid_idx] - prefix_sum[l])

            # Cost to change elements from mid_idx + 1 to r to target_val
            # These elements are greater than or equal to target_val
            # sum_{j=mid_idx+1}^{r} (nums[j] - target_val)
            # = sum_{j=mid_idx+1}^{r} nums[j] - (r - (mid_idx + 1) + 1) * target_val
            # = sum_{j=mid_idx+1}^{r} nums[j] - (r - mid_idx) * target_val
            # sum_{j=mid_idx+1}^{r} nums[j] = prefix_sum[r + 1] - prefix_sum[mid_idx + 1]
            right_cost = (prefix_sum[r + 1] - prefix_sum[mid_idx + 1]) - (r - mid_idx) * target_val

            return left_cost + right_cost

        # Function to check if frequency f is achievable
        def check(f: int) -> bool:
            # If f is 0 or negative, it's implicitly achievable (though BS range starts >= 1)
            if f <= 0: return True

            # If f > n, it's not possible
            if f > n: return False

            # Check if there's any contiguous subarray of length f
            # that can be made equal to its median with cost <= k.
            # The window slides from [l, r] where r = l + f - 1
            # l goes from 0 up to n - f
            for l in range(n - f + 1):
                r = l + f - 1
                cost = calculate_cost(l, r)
                if cost <= k:
                    return True
            return False

        # Binary search for the maximum frequency f
        # The possible maximum frequency ranges from 1 to n.
        low = 1
        high = n
        ans = 1 # Minimum possible frequency when n >= 1

        while low <= high:
            mid_f = low + (high - low) // 2 # Use integer division for safety

            if check(mid_f):
                # If frequency mid_f is achievable, it's a possible answer.
                # Try to achieve a higher frequency.
                ans = mid_f
                low = mid_f + 1
            else:
                # If frequency mid_f is not achievable, it's too high.
                # Try a lower frequency.
                high = mid_f - 1

        return ans