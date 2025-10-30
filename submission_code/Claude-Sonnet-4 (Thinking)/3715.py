class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Collect all interesting starting positions
        candidates = set()
        
        for l, r, c in coins:
            candidates.add(l - k + 1)  # Window starts overlapping segment
            candidates.add(l)          # Window's left edge aligns with segment's left edge
            candidates.add(r - k + 1)  # Window's right edge aligns with segment's right edge
            candidates.add(r + 1)      # Window stops overlapping segment
        
        max_coins = 0
        
        for start in candidates:
            # Calculate coins in window [start, start + k - 1]
            window_coins = 0
            for l, r, c in coins:
                # Find intersection of [start, start + k - 1] and [l, r]
                intersection_start = max(start, l)
                intersection_end = min(start + k - 1, r)
                if intersection_start <= intersection_end:
                    window_coins += (intersection_end - intersection_start + 1) * c
            
            max_coins = max(max_coins, window_coins)
        
        return max_coins