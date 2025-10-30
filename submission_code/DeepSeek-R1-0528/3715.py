from collections import defaultdict

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        events = defaultdict(int)
        critical_set = set()
        
        for l, r, c in coins:
            critical_set.add(l - k)
            critical_set.add(l)
            critical_set.add(r - k + 1)
            critical_set.add(r + 1)
            events[l - k] += c
            events[l] -= c
            events[r - k + 1] -= c
            events[r + 1] += c
        
        if not critical_set:
            return 0
        
        critical_list = sorted(critical_set)
        x0 = critical_list[0]
        total = 0
        for l, r, c in coins:
            low = max(l, x0)
            high = min(r, x0 + k - 1)
            if low <= high:
                total += (high - low + 1) * c
        
        max_total = total
        current_slope = events.get(x0, 0)
        current_x = x0
        current_F = total
        
        for i in range(1, len(critical_list)):
            next_x = critical_list[i]
            next_F = current_F + current_slope * (next_x - current_x)
            if next_F > max_total:
                max_total = next_F
            current_slope += events.get(next_x, 0)
            current_x = next_x
            current_F = next_F
        
        return max_total