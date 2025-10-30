from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        o0 = (n + 1) // 2
        e0 = n // 2
        available_odds = [i for i in range(1, n + 1) if i % 2 == 1]
        available_evens = [i for i in range(1, n + 1) if i % 2 == 0]
        
        dp = [[[0] * 3 for _ in range(e0 + 1)] for __ in range(o0 + 1)]
        
        for lp_index in range(3):
            dp[0][0][lp_index] = 1
        
        for o in range(o0 + 1):
            for e in range(e0 + 1):
                if o == 0 and e == 0:
                    continue
                for lp_val in [-1, 0, 1]:
                    idx = self.get_index(lp_val)
                    total = 0
                    if lp_val == -1:
                        if o > 0:
                            total += o * dp[o-1][e][self.get_index(1)]
                        if e > 0:
                            total += e * dp[o][e-1][self.get_index(0)]
                    elif lp_val == 1:
                        if e > 0:
                            total = e * dp[o][e-1][self.get_index(0)]
                    else:
                        if o > 0:
                            total = o * dp[o-1][e][self.get_index(1)]
                    
                    if total > k + 1:
                        total = k + 1
                    dp[o][e][idx] = total
        
        total_permutations = dp[o0][e0][self.get_index(-1)]
        if total_permutations < k:
            return []
        
        res = []
        o_curr = o0
        e_curr = e0
        last_parity = -1
        
        for _ in range(n):
            if last_parity == -1:
                i_odd, i_even = 0, 0
                candidates = []
                len_odds = len(available_odds)
                len_evens = len(available_evens)
                while i_odd < len_odds and i_even < len_evens:
                    if available_odds[i_odd] < available_evens[i_even]:
                        candidates.append(available_odds[i_odd])
                        i_odd += 1
                    else:
                        candidates.append(available_evens[i_even])
                        i_even += 1
                if i_odd < len_odds:
                    candidates.extend(available_odds[i_odd:])
                if i_even < len_evens:
                    candidates.extend(available_evens[i_even:])
            elif last_parity == 1:
                candidates = available_evens
            else:
                candidates = available_odds
            
            found = False
            for cand in candidates:
                if cand % 2 == 1:
                    new_o = o_curr - 1
                    new_e = e_curr
                    new_last_parity = 1
                else:
                    new_o = o_curr
                    new_e = e_curr - 1
                    new_last_parity = 0
                
                count_next = dp[new_o][new_e][self.get_index(new_last_parity)]
                if k <= count_next:
                    res.append(cand)
                    if cand % 2 == 1:
                        available_odds.remove(cand)
                    else:
                        available_evens.remove(cand)
                    o_curr = new_o
                    e_curr = new_e
                    last_parity = new_last_parity
                    found = True
                    break
                else:
                    k -= count_next
            
            if not found:
                return []
        
        return res
    
    def get_index(self, lp):
        if lp == -1:
            return 0
        elif lp == 0:
            return 1
        else:
            return 2