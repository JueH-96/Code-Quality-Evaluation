import math
from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_diff = float('inf')
        
        # Precompute next position for each bit and each index
        next_pos = [None] * 30
        for b in range(30):
            # Find positions where bit b is set
            pos_list = [i for i in range(n) if nums[i] & (1 << b)]
            # Add sentinel value n
            pos_list.append(n)
            # Compute next_b for each i
            next_b = [0] * n
            idx = 0
            len_p = len(pos_list)
            for i in range(n):
                while pos_list[idx] <= i:
                    idx += 1
                # Now pos_list[idx] > i
                next_b[i] = pos_list[idx]
            next_pos[b] = next_b
        
        # Now for each starting index l
        for l in range(n):
            current_r = l
            or_val = 0
            # Add the first element
            or_val |= nums[current_r]
            diff = abs(or_val - k)
            if diff < min_diff:
                min_diff = diff
            # Now find next changes where OR might change
            while True:
                # Find the minimum next j over bits not set in or_val
                next_r_candidate = n + 1  # Sentinel larger than n
                for b in range(30):
                    if (or_val & (1 << b)) == 0:  # Bit not set
                        next_j = next_pos[b][current_r]  # Next index > current_r with bit b set or n
                        if next_j < next_r_candidate:
                            next_r_candidate = next_j
                if next_r_candidate >= n:  # No valid next index or out of bounds
                    break
                # Move to the next candidate and add the element
                current_r = next_r_candidate
                or_val |= nums[current_r]
                # OR has changed, compute the difference
                diff = abs(or_val - k)
                if diff < min_diff:
                    min_diff = diff
        
        return min_diff