import bisect
from collections import defaultdict

class Solution:
    def minChanges(self, nums: list, k: int) -> int:
        n = len(nums)
        pairs = n // 2
        total_pairs = pairs
        
        d_to_Lps = defaultdict(list)
        L_p_list = []
        cnt_d = defaultdict(int)
        
        for i in range(pairs):
            a = nums[i]
            b = nums[n - 1 - i]
            d = abs(a - b)
            cnt_d[d] += 1
            
            min_val = min(a, b)
            max_val = max(a, b)
            L = max(max_val, (k - min_val)) + 1
            L_p_list.append(L)
            
            d_to_Lps[d].append(L)
        
        L_p_list.sort()
        
        for d in d_to_Lps:
            d_to_Lps[d].sort()
        
        min_changes = float('inf')
        
        for X in range(0, k + 1):
            cnt_ge = bisect.bisect_right(L_p_list, X)
            
            cnt_d_ge = 0
            if X in d_to_Lps:
                Lps = d_to_Lps[X]
                cnt_d_ge = bisect.bisect_right(Lps, X)
            
            current_sum = (total_pairs - cnt_d.get(X, 0)) + (cnt_ge - cnt_d_ge)
            
            if current_sum < min_changes:
                min_changes = current_sum
        
        return min_changes