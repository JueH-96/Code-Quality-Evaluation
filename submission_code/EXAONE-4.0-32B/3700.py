from collections import defaultdict
from typing import List

mod = 10**9 + 7

def nCr(n, r):
    if n < r:
        return 0
    if r == 0:
        return 1
    if r == 1:
        return n
    if r == 2:
        return (n * (n - 1) // 2) % mod
    return 0

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        
        for i in range(n):
            x = nums[i]
            left_side = nums[:i]
            right_side = nums[i+1:]
            
            left_count_x = left_side.count(x)
            right_count_x = right_side.count(x)
            left_non_x_count = len(left_side) - left_count_x
            right_non_x_count = len(right_side) - right_count_x
            
            for a in range(0, min(2, left_count_x) + 1):
                for b in range(0, min(2, right_count_x) + 1):
                    k = 1 + a + b
                    if k == 1:
                        continue
                    if a + b >= 2:
                        if 2 - a > left_non_x_count or 2 - b > right_non_x_count:
                            continue
                        ways_left_x = nCr(left_count_x, a)
                        ways_right_x = nCr(right_count_x, b)
                        ways_left_non_x = nCr(left_non_x_count, 2 - a)
                        ways_right_non_x = nCr(right_non_x_count, 2 - b)
                        total_ways = ways_left_x * ways_right_x % mod
                        total_ways = total_ways * ways_left_non_x % mod
                        total_ways = total_ways * ways_right_non_x % mod
                        ans = (ans + total_ways) % mod
                    else:
                        if 2 - a > left_non_x_count or 2 - b > right_non_x_count:
                            continue
                        left_freq = defaultdict(int)
                        for num in left_side:
                            if num != x:
                                left_freq[num] += 1
                        right_freq = defaultdict(int)
                        for num in right_side:
                            if num != x:
                                right_freq[num] += 1
                        
                        distinct_vals = set(left_freq.keys()) | set(right_freq.keys())
                        
                        dp = [[0]*3 for _ in range(3)]
                        dp[0][0] = 1
                        for val in distinct_vals:
                            next_dp = [row[:] for row in dp]
                            for i_val in range(3):
                                for j_val in range(3):
                                    if dp[i_val][j_val] == 0:
                                        continue
                                    if val in left_freq and val not in right_freq:
                                        if i_val < 2 - a:
                                            next_dp[i_val+1][j_val] = (next_dp[i_val+1][j_val] + dp[i_val][j_val] * left_freq[val]) % mod
                                    elif val in right_freq and val not in left_freq:
                                        if j_val < 2 - b:
                                            next_dp[i_val][j_val+1] = (next_dp[i_val][j_val+1] + dp[i_val][j_val] * right_freq[val]) % mod
                                    else:
                                        if i_val < 2 - a:
                                            next_dp[i_val+1][j_val] = (next_dp[i_val+1][j_val] + dp[i_val][j_val] * left_freq[val]) % mod
                                        if j_val < 2 - b:
                                            next_dp[i_val][j_val+1] = (next_dp[i_val][j_val+1] + dp[i_val][j_val] * right_freq[val]) % mod
                            dp = next_dp
                        ways_non_x = dp[2-a][2-b]
                        ways_left_x = nCr(left_count_x, a)
                        ways_right_x = nCr(right_count_x, b)
                        total_ways = ways_left_x * ways_right_x % mod
                        total_ways = total_ways * ways_non_x % mod
                        ans = (ans + total_ways) % mod
        return ans