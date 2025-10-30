import bisect
from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        coins.sort()
        n = len(coins)
        sum_r_c = [0] * (n + 1)
        sum_l_c = [0] * (n + 1)
        sum_c = [0] * (n + 1)
        l_list = []
        r_list = []
        
        for i in range(n):
            l, r, c = coins[i]
            sum_r_c[i+1] = sum_r_c[i] + r * c
            sum_l_c[i+1] = sum_l_c[i] + l * c
            sum_c[i+1] = sum_c[i] + c
            l_list.append(l)
            r_list.append(r)
        
        candidates = []
        for l, r, _ in coins:
            candidates.append(l)
            candidates.append(r + 1)
        
        max_total = 0
        
        for s in candidates:
            e = s + k - 1
            # Find right_end: largest index with l_i <= e
            right_end = bisect.bisect_right(l_list, e) - 1
            if right_end < 0:
                continue
            # Find left_start: first index in 0..right_end with r_i >= s
            left_start = bisect.bisect_left(r_list, s, 0, right_end + 1)
            if left_start > right_end:
                continue
            # Find split1: last index in left_start..right_end with r_i <= e
            split1 = bisect.bisect_right(r_list, e, left_start, right_end + 1) - 1
            if split1 < left_start:
                split1 = left_start - 1
            # Find split2: first index in left_start..right_end with l_i >= s
            split2 = bisect.bisect_left(l_list, s, left_start, right_end + 1)
            if split2 > right_end:
                split2 = right_end + 1
            # Calculate the sum
            sum_r_part = sum_r_c[split1 + 1] - sum_r_c[left_start]
            sum_e_part = e * (sum_c[right_end + 1] - sum_c[split1 + 1])
            sum_l_part = s * (sum_c[split2] - sum_c[left_start])
            sum_lc_part = sum_l_c[right_end + 1] - sum_l_c[split2]
            current_sum = (sum_r_part + sum_e_part) - (sum_l_part + sum_lc_part) + (sum_c[right_end + 1] - sum_c[left_start])
            if current_sum > max_total:
                max_total = current_sum
        
        return max_total