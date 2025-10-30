from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        events = []
        for li, ri, ci in coins:
            length = ri - li +1
            if length >=k:
                # Increasing overlap
                start_increase = li - k +1
                end_increase = li
                # Decreasing overlap
                start_decrease = ri - k +1
                end_decrease = ri +1
                events.append((start_increase, ci))
                events.append((end_increase, -ci))
                events.append((start_decrease, -ci))
                events.append((end_decrease, ci))
            else:
                # Overlap increases and then decreases
                start_increase = li - k +1
                end_increase = li
                end_decrease = ri +1
                events.append((start_increase, ci))
                events.append((end_increase, -ci))
                events.append((end_decrease, -ci))
        
        # Sort the events by S
        events.sort()
        
        current_sum = 0
        current_slope = 0
        max_sum = 0
        last_S = None
        
        for S, delta_slope in events:
            if last_S is not None:
                delta_S = S - last_S
                current_sum += current_slope * delta_S
                max_sum = max(max_sum, current_sum)
            current_slope += delta_slope
            last_S = S
        
        # After all events, there might be a remaining range
        max_sum = max(max_sum, current_sum)
        
        return max_sum