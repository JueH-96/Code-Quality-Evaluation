import collections
from bisect import bisect_left, bisect_right
from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Pre-processing:
        # 1. Create a sorted list of unique numbers and their counts (groups).
        counts = collections.Counter(nums)
        groups = sorted(counts.items())
        
        num_groups = len(groups)
        group_vals = [g[0] for g in groups]
        
        # 2. Create prefix sums of the counts for efficient range queries.
        prefix_counts = [0] * (num_groups + 1)
        for i in range(num_groups):
            prefix_counts[i+1] = prefix_counts[i] + groups[i][1]

        # 3. Keep a sorted version of the original array for Scenario 1 in check().
        sorted_nums = sorted(nums)

        def check(f: int) -> bool:
            if f == 0:
                return True
            if f > n:
                return False

            # Scenario 1: f <= numOperations
            # We can change all f elements. We only need to find a group S of size f
            # where max(S) - min(S) <= 2k.
            if f <= numOperations:
                for i in range(n - f + 1):
                    if sorted_nums[i+f-1] - sorted_nums[i] <= 2 * k:
                        return True
                return False

            # Scenario 2: f > numOperations
            # The target must be the mode `m` of the chosen group S.
            # Conditions:
            # 1. |s - m| <= k for all s in S.
            # 2. count(m in S) >= f - numOperations.
            min_mode_count = f - numOperations
            
            for i in range(num_groups):
                m, c = groups[i]
                
                # `m` is a candidate for the mode if its own count `c` is at least `min_mode_count`.
                if c >= min_mode_count:
                    # Check if we can find f elements in total within the range [m-k, m+k].
                    start_idx = bisect_left(group_vals, m - k)
                    end_idx = bisect_right(group_vals, m + k)
                    
                    total_available = prefix_counts[end_idx] - prefix_counts[start_idx]
                    
                    if total_available >= f:
                        return True
            
            return False

        # Binary search for the maximum possible frequency f.
        low, high = 0, n
        ans = 0
        while low <= high:
            mid = low + (high - low) // 2
            if mid == 0:
                low = mid + 1
                continue
            
            if check(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
                
        return ans