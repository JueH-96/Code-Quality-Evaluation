import bisect
from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute AND of all elements
        and_full = nums[0]
        for num in nums[1:]:
            and_full &= num
            if and_full == 0:
                break
        if and_full > 0:
            return 1
        
        # Precompute suffix AND
        suffix_and_arr = [0] * n
        suffix_and_arr[n - 1] = nums[n - 1]
        for i in range(n - 2, -1, -1):
            suffix_and_arr[i] = nums[i] & suffix_and_arr[i + 1]
        
        # Precompute positions where each bit is 0
        zero_pos = [[] for _ in range(20)]  # Bits 0 to 19
        for idx in range(n):
            num = nums[idx]
            for b in range(20):
                if (num & (1 << b)) == 0:  # Bit b is 0
                    zero_pos[b].append(idx)
        
        # Partition the array
        count = 0
        pos = 0  # Current start position
        while pos < n:
            # Compute j_min: max over b of min k >= pos with bit b=0
            j_min = -1
            for b in range(20):
                pos_list = zero_pos[b]
                idx_b = bisect.bisect_left(pos_list, pos)
                if idx_b < len(pos_list):
                    k_b = pos_list[idx_b]
                    if k_b >= pos:
                        if j_min == -1:
                            j_min = k_b
                        else:
                            j_min = max(j_min, k_b)
            
            j = j_min
            # Check if can end here
            if j >= n - 1 or (j < n - 1 and suffix_and_arr[j + 1] == 0):
                count += 1
                pos = j + 1
            else:
                # Increment j until condition is satisfied
                j += 1
                while j < n - 1 and suffix_and_arr[j + 1] != 0:
                    j += 1
                # After while, end subarray at j
                count += 1
                pos = j + 1
        
        return count