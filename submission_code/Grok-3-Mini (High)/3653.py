from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        # Compute prefix sums
        prefix = [0]
        cumsum = 0
        for num in nums:
            cumsum += num
            prefix.append(cumsum)
        
        # Group prefix sums by index % k
        groups = [[] for _ in range(k)]
        for i in range(len(prefix)):
            groups[i % k].append(prefix[i])
        
        # Initialize answer to negative infinity
        ans = float('-inf')
        
        # For each group
        for group in groups:
            if len(group) >= 2:
                min_so_far = group[0]
                max_diff_group = float('-inf')
                for j in range(1, len(group)):
                    diff = group[j] - min_so_far
                    if diff > max_diff_group:
                        max_diff_group = diff
                    # Update min_so_far after computing diff
                    min_so_far = min(min_so_far, group[j])
                # After processing the group, update ans
                ans = max(ans, max_diff_group)
        
        return int(ans)