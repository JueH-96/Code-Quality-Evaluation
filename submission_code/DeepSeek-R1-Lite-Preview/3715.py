from typing import List
from bisect import bisect_left

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Step 1: Create events for coin additions and subtractions
        events = []
        for segment in coins:
            l, r, c = segment
            events.append((l, c))        # Start of coin addition
            events.append((r + 1, -c))  # End of coin addition

        # Step 2: Sort the events by position
        events.sort(key=lambda x: x[0])
        
        # Step 3: Compute prefix sums at critical positions
        prefix_sums = {}
        prefix_sum = 0
        previous_position = 0
        current_coin = 0
        i = 0
        n = len(events)
        positions = []
        
        # Collect unique positions
        pos_set = set()
        for event in events:
            pos_set.add(event[0])
        positions = sorted(pos_set)
        
        # Compute prefix sums
        prefix_sums[0] = 0
        for pos in positions:
            if pos > previous_position:
                prefix_sum += (pos - previous_position) * current_coin
                prefix_sums[pos] = prefix_sum
            if i < n and events[i][0] == pos:
                current_coin += events[i][1]
                i += 1
            previous_position = pos
        
        # Step 4: Identify critical x positions
        critical_x = set()
        for pos in positions:
            critical_x.add(pos - k + 1)
            critical_x.add(pos)
        critical_x = sorted(critical_x)
        
        # Step 5: Find the maximum sum over all critical x positions
        max_sum = 0
        for x in critical_x:
            if x + k - 1 < 0:
                continue
            # Find prefix_sum[x + k - 1]
            pos = x + k - 1
            idx = bisect_left(positions, pos)
            if idx < len(positions) and positions[idx] == pos:
                sum_end = prefix_sums[pos]
            elif idx < len(positions):
                sum_end = prefix_sums[positions[idx - 1]] + (pos - positions[idx - 1]) * current_coin
            else:
                sum_end = prefix_sums[positions[-1]] + (pos - positions[-1]) * current_coin
            # Find prefix_sum[x - 1]
            pos_prev = x - 1
            if pos_prev < 0:
                sum_start = 0
            else:
                idx_prev = bisect_left(positions, pos_prev)
                if idx_prev < len(positions) and positions[idx_prev] == pos_prev:
                    sum_start = prefix_sums[pos_prev]
                elif idx_prev < len(positions):
                    sum_start = prefix_sums[positions[idx_prev - 1]] + (pos_prev - positions[idx_prev - 1]) * current_coin
                else:
                    sum_start = prefix_sums[positions[-1]] + (pos_prev - positions[-1]) * current_coin
            # Calculate Sum(x)
            sum_x = sum_end - sum_start
            if sum_x > max_sum:
                max_sum = sum_x
        return max_sum