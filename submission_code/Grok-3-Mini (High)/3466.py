import bisect
from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans = 0
        n = len(nums)
        i = 0
        while i < n:
            if (nums[i] & k) == k:
                start = i
                j = i
                while j < n and (nums[j] & k) == k:
                    j += 1
                end = j - 1
                seg_count = self.count_in_segment(nums, start, end, k)
                ans += seg_count
                i = j
            else:
                i += 1
        return ans
    
    def count_in_segment(self, nums, start, end, k):
        S_bits = [b for b in range(30) if (k & (1 << b)) == 0]
        len_S = len(S_bits)
        
        if len_S == 0:
            len_seg = end - start + 1
            return (len_seg * (len_seg + 1)) // 2
        
        # Compute zeros for each bit and check if all have at least one zero
        zeros_dict = {}
        all_have_zero = True
        for b in S_bits:
            zeros = [idx for idx in range(start, end + 1) if (nums[idx] & (1 << b)) == 0]
            if not zeros:
                all_have_zero = False
                break  # No need to compute further, return 0
            zeros_dict[b] = zeros  # zeros is already sorted since range is sequential
        
        if not all_have_zero:
            return 0
        
        # Now all bits have at least one zero in the segment
        count = 0
        for left in range(start, end + 1):
            max_next_j = -1
            for b in S_bits:
                zeros_b = zeros_dict[b]
                idx = bisect.bisect_left(zeros_b, left)
                if idx < len(zeros_b):
                    next_j_b = zeros_b[idx]
                else:
                    next_j_b = end + 1  # Beyond the end of segment
                if next_j_b > max_next_j:
                    max_next_j = next_j_b
            right_min_left = max_next_j
            if right_min_left <= end:
                num_right = end - right_min_left + 1
                count += num_right
        
        return count