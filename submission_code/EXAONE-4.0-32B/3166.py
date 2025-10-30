import math
from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        freq_counter = Counter(nums)
        freqs = list(freq_counter.values())
        sorted_freqs = sorted(freqs, reverse=True)
        max_f = max(freqs) if freqs else 0
        
        freq_count = [0] * (max_f + 1)
        for f in freqs:
            if f <= max_f:
                freq_count[f] += 1
        
        k = 1
        ans = n
        while k <= n:
            t = n // k
            if t == 0:
                k_next = n + 1
            else:
                k_next = n // t + 1
            
            L = t + 1
            S_val = 0
            for f_val in range(1, max_f + 1):
                if freq_count[f_val] > 0:
                    ceil_val = (f_val + L - 1) // L
                    S_val += freq_count[f_val] * ceil_val
            
            start_j = max(k, S_val)
            end_j = min(k_next, n + 1)
            for j in range(start_j, end_j):
                avail_large = n - j * t
                avail_small = j * (t + 1) - n
                if avail_large < 0 or avail_small < 0:
                    continue
                
                temp_large = avail_large
                temp_small = avail_small
                valid = True
                for f_val in sorted_freqs:
                    r0 = f_val % t
                    num = f_val - temp_small * t
                    if num <= 0:
                        lb = 0
                    else:
                        lb = (num + t) // (t + 1)
                    
                    x0 = max(lb, r0)
                    a = x0 % t
                    diff = (r0 - a) % t
                    candidate_x = x0 + diff
                    
                    if candidate_x > temp_large or candidate_x * (t + 1) > f_val:
                        valid = False
                        break
                    
                    remainder = f_val - candidate_x * (t + 1)
                    if remainder < 0:
                        valid = False
                        break
                    if remainder % t != 0:
                        valid = False
                        break
                    y = remainder // t
                    if y > temp_small:
                        valid = False
                        break
                    
                    temp_large -= candidate_x
                    temp_small -= y
                
                if valid:
                    return j
            
            k = k_next
        
        return n