import math
from typing import List

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        left_freq = [0] * 1001
        ans = 0
        
        for q in range(n):
            if q >= 2:
                p_val = nums[q-2]
                if 1 <= p_val <= 1000:
                    left_freq[p_val] += 1
            
            right_freq = [0] * 1001
            
            for r in range(n-3, q+1, -1):
                s_index = r + 2
                s_val = nums[s_index]
                if 1 <= s_val <= 1000:
                    right_freq[s_val] += 1
                
                A = nums[q]
                B = nums[r]
                total_temp = 0
                
                if B == 1:
                    for s_val in range(1, 1001):
                        cnt = right_freq[s_val]
                        if cnt == 0:
                            continue
                        target_p = A * s_val
                        if target_p > 1000:
                            break
                        if target_p < 1:
                            continue
                        total_temp += left_freq[target_p] * cnt
                else:
                    g = math.gcd(A, B)
                    A1 = A // g
                    B1 = B // g
                    k = 1
                    while True:
                        s_val = k * B1
                        if s_val > 1000:
                            break
                        target_p = A1 * k
                        if target_p > 1000:
                            break
                        cnt = right_freq[s_val]
                        if cnt:
                            total_temp += left_freq[target_p] * cnt
                        k += 1
                
                ans += total_temp
        
        return ans