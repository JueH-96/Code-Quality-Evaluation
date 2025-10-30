class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        events = []
        for l, r, c in coins:
            events.append((l, c))
            events.append((r + 1, -c))
        
        events.sort()
        
        current_coins = 0
        coin_map = {}
        
        for pos, coin_change in events:
            current_coins += coin_change
            coin_map[pos] = current_coins
        
        positions = sorted(coin_map.keys())
        
        max_coins = 0
        
        for i in range(len(positions)):
            start = positions[i]
            end = start + k - 1
            
            total_coins = 0
            
            for l, r, c in coins:
                
                intersection_start = max(start, l)
                intersection_end = min(end, r)
                
                if intersection_start <= intersection_end:
                    total_coins += (intersection_end - intersection_start + 1) * c
            
            max_coins = max(max_coins, total_coins)
            
        return max_coins