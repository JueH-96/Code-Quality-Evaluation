class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        n = len(coins)
        events = []
        for l, r, c in coins:
            events.append((l, c))
            events.append((r + 1, -c))
        
        events.sort()
        
        ans = 0
        curr_sum = 0
        window = []
        
        last_pos = 0
        
        for pos, coin in events:
            
            if pos > last_pos:
                length = pos - last_pos
                
                for _ in range(length):
                    window.append(curr_sum)
                    if len(window) > k:
                        window.pop(0)
                
                if len(window) == k:
                    ans = max(ans, sum(window))
            
            curr_sum += coin
            last_pos = pos
            
        return ans