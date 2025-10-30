class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort coins by left boundary
        coins.sort()
        
        max_coins = 0
        
        # Consider all potential starting positions
        positions = set()
        
        for l, r, c in coins:
            positions.add(l)  # Start of segment
            positions.add(max(1, l - k + 1))  # k positions before start
            if r >= k:
                positions.add(r - k + 1)  # k positions before end
        
        # For each potential starting position
        for start in positions:
            end = start + k - 1
            current_sum = 0
            
            # Calculate sum for this k-window
            for l, r, c in coins:
                if r < start or l > end:
                    # Segment doesn't overlap with window
                    continue
                
                # Calculate overlap
                overlap_start = max(start, l)
                overlap_end = min(end, r)
                
                if overlap_start <= overlap_end:
                    current_sum += (overlap_end - overlap_start + 1) * c
            
            max_coins = max(max_coins, current_sum)
        
        return max_coins