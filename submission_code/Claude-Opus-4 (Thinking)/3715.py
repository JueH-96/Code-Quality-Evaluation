class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        positions = []
        
        # Generate candidate starting positions
        for l, r, c in coins:
            positions.append(l)  # Start at beginning of segment
            if r - k + 1 > 0:    # Window ends at end of segment
                positions.append(r - k + 1)
        
        # Remove duplicates
        positions = list(set(positions))
        
        max_coins = 0
        
        # Check each candidate position
        for start in positions:
            end = start + k - 1
            total = 0
            
            # Calculate coins for this k-window
            for l, r, c in coins:
                # Calculate overlap between [start, end] and [l, r]
                overlap_start = max(start, l)
                overlap_end = min(end, r)
                
                if overlap_start <= overlap_end:
                    overlap_length = overlap_end - overlap_start + 1
                    total += overlap_length * c
            
            max_coins = max(max_coins, total)
        
        return max_coins