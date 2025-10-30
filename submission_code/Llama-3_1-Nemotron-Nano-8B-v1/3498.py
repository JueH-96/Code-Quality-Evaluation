from typing import List
from collections import defaultdict

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n // 2
        sum_0 = defaultdict(int)
        delta = [0] * (k + 2)  # delta array to track interval coverage
        
        for i in range(m):
            a = nums[i]
            b = nums[n - 1 - i]
            original_x = abs(a - b)
            sum_0[original_x] += 1
            
            # Calculate max possible values when changing a or b
            max_a = max(a, k - a)
            max_b = max(b, k - b)
            current_max = max(max_a, max_b)
            
            # Update delta array for the interval [0, current_max]
            delta[0] += 1
            if current_max + 1 <= k:
                delta[current_max + 1] -= 1
        
        # Compute prefix sums to get count_intervals
        count_intervals = [0] * (k + 1)
        current = 0
        for x in range(k + 1):
            current += delta[x]
            count_intervals[x] = current
        
        min_changes = float('inf')
        for x in range(k + 1):
            sum_0_x = sum_0.get(x, 0)
            sum_1_x = count_intervals[x] - sum_0_x
            sum_2_x = (m - sum_0_x - sum_1_x) if x <= k else 0
            
            if sum_0_x + sum_1_x + sum_2_x != m:
                continue  # Skip if not all pairs can achieve X
            
            total = sum_1_x + 2 * sum_2_x
            if total < min_changes:
                min_changes = total
        
        return min_changes if min_changes != float('inf') else -1