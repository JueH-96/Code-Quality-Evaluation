from typing import List

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        # maximum possible value we have to care about
        max_g = max(groups)
        
        # Step-1: for every element value keep the first (smallest) index
        first_idx = {}
        for idx, val in enumerate(elements):
            if val not in first_idx:          # the first time we meet val → smallest index
                first_idx[val] = idx
        
        # Step-2: sieve-like propagation.
        # min_div[m] will store the minimal element-index that divides m (∞ = none yet)
        INF = 10**9
        min_div = [INF] * (max_g + 1)
        
        for val, idx in first_idx.items():
            if val > max_g:                # cannot divide any group value
                continue
            # push this index to all multiples of val
            for multiple in range(val, max_g + 1, val):
                if idx < min_div[multiple]:
                    min_div[multiple] = idx
        
        # Step-3: build the answer
        res = []
        for g in groups:
            if min_div[g] == INF:
                res.append(-1)
            else:
                res.append(min_div[g])
                
        return res