import math
from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # Compute prefix sum
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        S = prefix[-1]  # sum of nums
        # Values for cumulative sums at positions 0 to n-1
        vals = prefix[:n]
        # Create a map from value to index
        idx_map = {val: idx for idx, val in enumerate(vals)}
        # Compute min_k and max_k
        min_k_num = target - S + 1
        min_k = - ((-min_k_num) // S)  # ceil division
        max_k = (target + S - 1) // S  # floor division
        # k range from max(0, min_k) to max_k
        k_start = max(0, min_k)
        k_end = max_k
        min_length = float('inf')
        for k in range(k_start, k_end + 1):
            delta = target - k * S
            for q in range(n):
                req_val = vals[q] + delta
                if req_val in idx_map:
                    p = idx_map[req_val]
                    length_candidate = k * n + (p - q)
                    if length_candidate > 0 and length_candidate < min_length:
                        min_length = length_candidate
        if min_length == float('inf'):
            return -1
        else:
            return int(min_length)