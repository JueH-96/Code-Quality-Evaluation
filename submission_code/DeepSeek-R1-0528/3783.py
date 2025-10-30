class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        no = (n + 1) // 2
        ne = n // 2
        
        dp = [[[0]*2 for _ in range(ne+1)] for __ in range(no+1)]
        
        for o in range(0, no+1):
            for e in range(0, ne+1):
                for p in [0, 1]:
                    if o == 0 and e == 0:
                        dp[o][e][p] = 1
                    else:
                        if p == 1 and o > 0:
                            dp[o][e][p] = o * dp[o-1][e][0]
                        elif p == 0 and e > 0:
                            dp[o][e][p] = e * dp[o][e-1][1]
        
        total_permutations = dp[no][ne][0] + dp[no][ne][1]
        if k > total_permutations:
            return []
        
        odds = [i for i in range(1, n+1) if i % 2 == 1]
        evens = [i for i in range(1, n+1) if i % 2 == 0]
        o_count = no
        e_count = ne
        next_parity = None
        res = []
        
        for _ in range(n):
            if next_parity is None:
                available = sorted(odds + evens)
                found = False
                for candidate in available:
                    if candidate % 2 == 1:
                        if o_count == 0:
                            count_branch = 0
                        else:
                            count_branch = dp[o_count-1][e_count][0]
                    else:
                        if e_count == 0:
                            count_branch = 0
                        else:
                            count_branch = dp[o_count][e_count-1][1]
                    
                    if count_branch == 0:
                        continue
                    
                    if k <= count_branch:
                        res.append(candidate)
                        if candidate % 2 == 1:
                            odds.remove(candidate)
                            o_count -= 1
                            next_parity = 0
                        else:
                            evens.remove(candidate)
                            e_count -= 1
                            next_parity = 1
                        found = True
                        break
                    else:
                        k -= count_branch
                if not found:
                    return []
            else:
                if next_parity == 1:
                    if o_count == 0:
                        return []
                    available = odds
                    count_per_candidate = dp[o_count-1][e_count][0]
                else:
                    if e_count == 0:
                        return []
                    available = evens
                    count_per_candidate = dp[o_count][e_count-1][1]
                
                if count_per_candidate == 0:
                    return []
                
                total_branch = len(available) * count_per_candidate
                if k > total_branch:
                    return []
                
                index = (k - 1) // count_per_candidate
                candidate = available[index]
                res.append(candidate)
                if next_parity == 1:
                    del available[index]
                    o_count -= 1
                    next_parity = 0
                else:
                    del available[index]
                    e_count -= 1
                    next_parity = 1
                k -= index * count_per_candidate
        
        return res