from collections import defaultdict

class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        points = {
            ('F', 'F'): (0, 0),
            ('F', 'W'): (0, 1),
            ('F', 'E'): (1, 0),
            ('W', 'F'): (1, 0),
            ('W', 'W'): (0, 0),
            ('W', 'E'): (0, 1),
            ('E', 'F'): (0, 1),
            ('E', 'W'): (1, 0),
            ('E', 'E'): (0, 0),
        }
        
        current = defaultdict(int)
        current[(None, 0)] = 1  # Initial state: no previous move, difference 0
        
        for i in range(len(s)):
            a = s[i]
            next_state = defaultdict(int)
            for (last_b, diff), cnt in current.items():
                for next_b in ['F', 'W', 'E']:
                    if last_b is not None and next_b == last_b:
                        continue
                    key = (a, next_b)
                    a_p, b_p = points[key]
                    new_diff = diff + (b_p - a_p)
                    next_state[(next_b, new_diff)] = (next_state[(next_b, new_diff)] + cnt) % MOD
            current = next_state
        
        total = 0
        for (last_b, diff), cnt in current.items():
            if diff > 0:
                total = (total + cnt) % MOD
        return total % MOD