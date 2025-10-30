class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Create a list of positions to check
        positions_to_check = set()
        for l, r, _ in coins:
            # Window starts at the beginning of the segment
            positions_to_check.add(l)
            
            # Window ends at the end of the segment
            end_start = r - k + 1
            if end_start >= 1:  # Ensure the window doesn't start before position 1
                positions_to_check.add(end_start)
        
        # Try each position as the start of a window of size k
        max_coins = 0
        for start in positions_to_check:
            end = start + k - 1
            
            total_coins = 0
            for l, r, c in coins:
                # If the segment is completely outside the window, skip it
                if r < start or l > end:
                    continue
                
                # If the segment overlaps with the window, add the overlapping part's coins
                overlap_start = max(start, l)
                overlap_end = min(end, r)
                total_coins += (overlap_end - overlap_start + 1) * c
            
            max_coins = max(max_coins, total_coins)
        
        return max_coins