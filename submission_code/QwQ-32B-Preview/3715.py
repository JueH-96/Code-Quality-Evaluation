from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        import bisect

        # Sort the segments by their start positions
        segs = sorted(coins, key=lambda x: x[0])
        
        # Collect critical points: starts and ends of segments
        critical_points = sorted(set([l for l, r, c in coins] + [r+1 for l, r, c in coins]))
        
        # Initialize pointers for the segments
        i = 0
        j = 0
        max_sum = 0
        
        # Iterate through each critical point as the start of the window
        for start in critical_points:
            end = start + k - 1
            
            # Move pointer j to include segments that start before or at end
            while j < len(segs) and segs[j][0] <= end:
                j += 1
            
            # Move pointer i to exclude segments that end before start
            while i < j and segs[i][1] < start:
                i += 1
            
            # Calculate the sum for the current window
            current_sum = 0
            for idx in range(i, j):
                l, r, c = segs[idx]
                overlap_start = max(start, l)
                overlap_end = min(end, r)
                overlap_length = overlap_end - overlap_start + 1
                current_sum += c * overlap_length
            
            # Update the maximum sum
            if current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum