class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        n = len(s)
        total_cost = 0
        for i in range(n):
            ord_s = ord(s[i]) - ord('a')
            ord_t = ord(t[i]) - ord('a')
            
            diff = ord_t - ord_s
            
            if diff == 0:
                continue
            
            if diff > 0:
                next_moves = diff
                prev_moves = 26 - diff
            else:
                next_moves = 26 + diff
                prev_moves = -diff
            
            next_cost_val = 0
            prev_cost_val = 0
            
            cost_next = 0
            for j in range(next_moves):
                idx = (ord_s + j) % 26
                cost_next += nextCost[idx]
                
            cost_prev = 0
            for j in range(prev_moves):
                idx = (ord_s - j - 1) % 26
                if idx < 0:
                    idx += 26
                cost_prev += previousCost[idx]
            
            total_cost += min(cost_next, cost_prev)
            
        return total_cost