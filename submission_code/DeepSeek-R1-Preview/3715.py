import bisect

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        coins.sort()
        n = len(coins)
        critical = set()
        for l, r, c in coins:
            critical.add(l)
            critical.add(r)
            critical.add(l - k + 1)
            critical.add(r - k + 1)
        critical = sorted(critical)
        max_sum = 0
        l_list = [x[0] for x in coins]
        r_list = [x[1] for x in coins]
        c_list = [x[2] for x in coins]
        
        for x in critical:
            a = x
            b = x + k - 1
            if b < a:
                continue  # invalid case when k is 0, but k >=1
            # Find coins with l_i <= b
            idx_b = bisect.bisect_right(l_list, b)
            if idx_b == 0:
                continue
            # Find the first r_i >= a in coins[0..idx_b-1]
            left = bisect.bisect_left(r_list, a, 0, idx_b)
            current_sum = 0
            for i in range(left, idx_b):
                l_i = coins[i][0]
                r_i = coins[i][1]
                c_i = coins[i][2]
                overlap_start = max(l_i, a)
                overlap_end = min(r_i, b)
                if overlap_start > overlap_end:
                    continue
                current_sum += (overlap_end - overlap_start + 1) * c_i
            if current_sum > max_sum:
                max_sum = current_sum
        return max_sum