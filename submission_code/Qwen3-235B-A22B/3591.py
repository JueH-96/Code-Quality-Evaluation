class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Precompute forward and backward prefix sums
        forward_prefix = [[0] * 26 for _ in range(26)]
        for a in range(26):
            total = 0
            for k in range(1, 26):  # k ranges from 1 to 25 (steps)
                idx = (a + k - 1) % 26
                total += nextCost[idx]
                forward_prefix[a][k] = total
        
        backward_prefix = [[0] * 26 for _ in range(26)]
        for a in range(26):
            total = 0
            for k in range(1, 26):  # k ranges from 1 to 25 (steps)
                idx = (a - (k - 1)) % 26
                total += previousCost[idx]
                backward_prefix[a][k] = total
        
        total_cost = 0
        for sc, tc in zip(s, t):
            if sc == tc:
                continue
            s_idx = ord(sc) - ord('a')
            t_idx = ord(tc) - ord('a')
            
            forward_diff = (t_idx - s_idx) % 26
            cost_forward = forward_prefix[s_idx][forward_diff]
            
            backward_diff = (s_idx - t_idx) % 26
            cost_backward = backward_prefix[s_idx][backward_diff]
            
            total_cost += min(cost_forward, cost_backward)
        
        return total_cost