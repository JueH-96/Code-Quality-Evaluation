import math
import bisect
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        MAX_NUM = 50000
        
        # Compute smallest prime factor (spf) array
        spf = [0] * (MAX_NUM + 1)
        for i in range(2, MAX_NUM + 1):
            if spf[i] == 0:
                for j in range(i, MAX_NUM + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
        
        # Compute Möbius function (mu) array using spf
        mu_arr = [0] * (MAX_NUM + 1)
        mu_arr[1] = 1
        for i in range(2, MAX_NUM + 1):
            temp_num = i
            mu_i = 1
            has_square_factor = False
            while temp_num > 1 and not has_square_factor:
                p = spf[temp_num]
                exp = 0
                while temp_num % p == 0:
                    temp_num //= p
                    exp += 1
                if exp > 1:
                    has_square_factor = True
                else:
                    mu_i *= -1
            if has_square_factor:
                mu_arr[i] = 0
            else:
                mu_arr[i] = mu_i
        
        # Compute count_div: number of elements divisible by each d
        count_div = [0] * (MAX_NUM + 1)
        for num in nums:
            for i in range(1, int(math.sqrt(num)) + 1):
                if num % i == 0:
                    d1 = i
                    d2 = num // i
                    count_div[d1] += 1
                    if d1 != d2:
                        count_div[d2] += 1  # Increment only if d1 and d2 are different
        
        # Compute G[d]: number of pairs with GCD exactly d using Möbius inversion
        G = [0] * (MAX_NUM + 1)
        for d in range(1, MAX_NUM + 1):
            G_d_sum = 0
            max_k = MAX_NUM // d
            for k in range(1, max_k + 1):
                m = d * k
                H_m = (count_div[m] * (count_div[m] - 1)) // 2  # comb(count_div[m], 2)
                G_d_sum += mu_arr[k] * H_m
            G[d] = G_d_sum
        
        # Build vals and freq_list for non-zero G[d]
        vals = []
        freq_list = []
        for d in range(1, MAX_NUM + 1):
            if G[d] > 0:
                vals.append(d)
                freq_list.append(G[d])
        
        # Compute prefix sum for cumulative frequencies
        prefix_sum = [0]
        for f in freq_list:
            prefix_sum.append(prefix_sum[-1] + f)
        
        # upper_bounds for binary search
        upper_bounds = prefix_sum[1:]  # prefix_sum[1:] corresponds to cumulative sums starting from index 1
        
        # Answer the queries using binary search
        answer = []
        for idx in queries:
            pos = bisect.bisect_right(upper_bounds, idx)
            ans_val = vals[pos]
            answer.append(ans_val)
        
        return answer