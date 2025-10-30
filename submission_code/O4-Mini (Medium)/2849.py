from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        # total number of subarrays
        total_sub = n * (n + 1) // 2
        
        maxv = max(nums)
        # no_v[v] = number of subarrays that do NOT contain value v
        no_v = [0] * (maxv + 2)
        
        # compute no_v for each v in [1..maxv]
        for v in range(1, maxv + 1):
            seg_len = 0
            cnt = 0
            for x in nums:
                if x == v:
                    # end of a segment without v
                    cnt += seg_len * (seg_len + 1) // 2
                    seg_len = 0
                else:
                    seg_len += 1
            # tail segment
            cnt += seg_len * (seg_len + 1) // 2
            no_v[v] = cnt
        
        # sum of unique counts over all subarrays
        # = sum_{v} (#subarrays that contain v)
        # = sum_{v}( total_sub - no_v[v] )
        sum_unique = 0
        for v in range(1, maxv + 1):
            sum_unique += (total_sub - no_v[v])
        
        # Now compute adjacency contributions:
        # For each x from 1..maxv-1,
        # adjacency_count[x] = #subarrays containing BOTH x and x+1
        # = total_sub - no_v[x] - no_v[x+1] + no_both(x)
        # where no_both(x) = #subarrays containing NEITHER x nor x+1
        sum_adj = 0
        for x in range(1, maxv):
            # count subarrays without x and without x+1
            seg_len = 0
            cnt_both = 0
            for y in nums:
                if y == x or y == x + 1:
                    cnt_both += seg_len * (seg_len + 1) // 2
                    seg_len = 0
                else:
                    seg_len += 1
            cnt_both += seg_len * (seg_len + 1) // 2
            
            adj_cnt = total_sub - no_v[x] - no_v[x+1] + cnt_both
            sum_adj += adj_cnt
        
        # sum of (unique_count - 1) over all subarrays
        sum_unique_minus1 = sum_unique - total_sub
        # sum of imbalance numbers = sum(unique_count - 1) - sum_adj
        return sum_unique_minus1 - sum_adj