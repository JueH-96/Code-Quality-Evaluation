class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        delta_map = {
            ('F', 'E'): -1,
            ('E', 'F'): 1,
            ('W', 'F'): -1,
            ('F', 'W'): 1,
            ('E', 'W'): -1,
            ('W', 'E'): 1,
            ('F', 'F'): 0,
            ('W', 'W'): 0,
            ('E', 'E'): 0,
        }
        
        from collections import defaultdict
        
        current_dp = defaultdict(int)
        current_dp[(None, 0)] = 1  # Initial state: prev is None, diff 0
        
        n = len(s)
        for i in range(n):
            A = s[i]
            next_dp = defaultdict(int)
            for (prev, diff), cnt in current_dp.items():
                for B in ['F', 'W', 'E']:
                    if prev is not None and B == prev:
                        continue
                    delta = delta_map[(A, B)]
                    new_diff = diff + delta
                    next_dp[(B, new_diff)] = (next_dp[(B, new_diff)] + cnt) % MOD
            current_dp = next_dp
        
        total = 0
        for (prev, diff), cnt in current_dp.items():
            if diff > 0:
                total = (total + cnt) % MOD
        return total