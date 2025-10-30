class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        def compute_max(scenario):
            current_sum = 0
            count_neg = 0
            max_val = -float('inf')
            for c in s:
                contrib, is_neg = scenario(c)
                current_sum += contrib
                if is_neg:
                    count_neg += 1
                possible = current_sum + 2 * min(count_neg, k)
                if possible > max_val:
                    max_val = possible
            return max_val if max_val != -float('inf') else 0
        
        scenarios = [
            lambda c: (1, False) if c in {'N', 'E'} else (-1, True),
            lambda c: (1, False) if c in {'E', 'S'} else (-1, True),
            lambda c: (1, False) if c in {'N', 'W'} else (-1, True),
            lambda c: (1, False) if c in {'S', 'W'} else (-1, True),
        ]
        
        max_dist = 0
        for scenario in scenarios:
            current = compute_max(scenario)
            if current > max_dist:
                max_dist = current
        return max_dist