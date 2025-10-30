from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Precompute next_sum and prev_sum for all possible shifts
        next_sum = [[0] * 26 for _ in range(26)]
        prev_sum = [[0] * 26 for _ in range(26)]
        
        for j in range(26):
            for k in range(1, 26):
                next_sum[j][k] = next_sum[j][k-1] + nextCost[(j + k - 1) % 26]
                prev_sum[j][k] = prev_sum[j][k-1] + previousCost[(j - k + 1) % 26]
        
        total = 0
        for sc, tc in zip(s, t):
            if sc == tc:
                continue
            s_pos = ord(sc) - ord('a')
            t_pos = ord(tc) - ord('a')
            f_steps = (t_pos - s_pos) % 26
            b_steps = (s_pos - t_pos) % 26
            cost_forward = next_sum[s_pos][f_steps]
            cost_backward = prev_sum[s_pos][b_steps]
            total += min(cost_forward, cost_backward)
        return total