class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort segments by left position
        coins.sort()
        
        # Collect all critical points where we might want to start a window
        critical_points = set()
        
        for left, right, _ in coins:
            # Window starting at segment start
            critical_points.add(left)
            # Window ending at segment end
            critical_points.add(right - k + 1)
            # Window starting right after segment
            critical_points.add(right + 1)
            # Window ending right before segment
            critical_points.add(left - k)
        
        max_coins = 0
        
        for start_pos in critical_points:
            if start_pos < 1:
                continue
                
            window_end = start_pos + k - 1
            current_coins = 0
            
            # Calculate coins in window [start_pos, window_end]
            for left, right, coin_count in coins:
                # Find intersection
                intersection_left = max(left, start_pos)
                intersection_right = min(right, window_end)
                
                if intersection_left <= intersection_right:
                    intersection_length = intersection_right - intersection_left + 1
                    current_coins += intersection_length * coin_count
            
            max_coins = max(max_coins, current_coins)
        
        return max_coins