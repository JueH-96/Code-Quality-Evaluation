import bisect
from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort coins by their starting coordinate
        coins.sort()
        n = len(coins)
        if n == 0:
            return 0
        
        # Extract lists for l, r, and c values
        l_list = [l for l, r, c in coins]
        r_list = [r for l, r, c in coins]
        c_list = [c for l, r, c in coins]
        
        # Generate critical points
        critical = {1}  # Start of the number line
        for l, r, c in coins:
            critical.add(l)
            critical.add(r)
            critical.add(l - 1)
            critical.add(r + 1)
        if coins:
            critical.add(r_list[-1] + 1)  # End of the last segment's gap
        critical_points = sorted(critical)
        
        # Precompute prefix sums
        pre_c = [0] * (n + 1)
        pre_c_l = [0] * (n + 1)
        pre_c_r = [0] * (n + 1)
        for i in range(n):
            pre_c[i+1] = pre_c[i] + c_list[i]
            pre_c_l[i+1] = pre_c_l[i] + c_list[i] * l_list[i]
            pre_c_r[i+1] = pre_c_r[i] + c_list[i] * r_list[i]
        
        max_sum = 0
        
        # Check segments with length >= k
        for i in range(n):
            length = r_list[i] - l_list[i] + 1
            if length >= k:
                max_sum = max(max_sum, c_list[i] * k)
        
        # Function to compute sum of coins in the window [a, b]
        def compute_sum(a: int, b: int) -> int:
            nonlocal l_list, r_list, pre_c, pre_c_l, pre_c_r, n
            # Find overlapping segments [low, high]
            high = bisect.bisect_right(l_list, b) - 1
            if high < 0:
                return 0
            # Find first index in [0, high] where r >= a
            low = bisect.bisect_left(r_list, a, 0, high + 1)
            if low > high:
                return 0
            
            # Calculate sum_c_total
            sum_c_total = pre_c[high + 1] - pre_c[low]
            
            # Calculate sum_c_min_r
            split_r = bisect.bisect_right(r_list, b, low, high + 1) - 1
            if split_r < low:
                sum_c_min_r_part1 = 0
                sum_c_min_r_part2 = b * (pre_c[high + 1] - pre_c[low])
            else:
                sum_c_min_r_part1 = pre_c_r[split_r + 1] - pre_c_r[low]
                sum_c_min_r_part2 = b * (pre_c[high + 1] - pre_c[split_r + 1])
            sum_c_min_r = sum_c_min_r_part1 + sum_c_min_r_part2
            sum1 = sum_c_min_r + sum_c_total
            
            # Calculate sum_c_max_l
            split_l = bisect.bisect_left(l_list, a, low, high + 1)
            sum_c_max_l_part1 = a * (pre_c[split_l] - pre_c[low])
            sum_c_max_l_part2 = (pre_c_l[high + 1] - pre_c_l[split_l])
            sum2 = sum_c_max_l_part1 + sum_c_max_l_part2
            
            return max(sum1 - sum2, 0)
        
        # Generate candidate windows starting at each critical point
        for s in critical_points:
            if s < 1:
                continue
            a = s
            b = s + k - 1
            current_sum = compute_sum(a, b)
            if current_sum > max_sum:
                max_sum = current_sum
        
        # Generate candidate windows ending at each critical point
        for e in critical_points:
            a = e - k + 1
            if a < 1:
                continue
            current_sum = compute_sum(a, e)
            if current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum