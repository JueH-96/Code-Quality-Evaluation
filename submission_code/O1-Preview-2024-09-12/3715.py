class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        events = []
        for l_i, r_i, c_i in coins:
            events.append((l_i, c_i))
            events.append((r_i + 1, -c_i))
        events.sort()
        
        positions = []
        for pos, _ in events:
            positions.append(pos)
        positions = sorted(set(positions))
        
        # Build intervals with constant coin counts
        curr_coins = 0
        prev_pos = None
        i = 0
        intervals = []
        while i < len(events):
            pos = events[i][0]
            delta = 0
            while i < len(events) and events[i][0] == pos:
                delta += events[i][1]
                i += 1
            if prev_pos is not None and pos > prev_pos:
                length = pos - prev_pos
                intervals.append((prev_pos, pos - 1, length, curr_coins))
            curr_coins += delta
            prev_pos = pos
        # Handle any remaining interval after the last event
        # Not needed since positions are finite in events
        
        # Sliding window over intervals
        left = 0
        curr_total_length = 0
        curr_total_coins = 0
        max_total_coins = 0
        for right in range(len(intervals)):
            interval_length = intervals[right][2]
            interval_coins = intervals[right][3]
            curr_total_length += interval_length
            curr_total_coins += interval_length * interval_coins
            
            # Shrink window from the left if total length exceeds k
            while curr_total_length > k:
                excess_length = curr_total_length - k
                left_interval_length = intervals[left][2]
                left_interval_coins = intervals[left][3]
                if left_interval_length <= excess_length:
                    curr_total_length -= left_interval_length
                    curr_total_coins -= left_interval_length * left_interval_coins
                    left += 1
                else:
                    # Remove part of the left interval
                    curr_total_length -= excess_length
                    curr_total_coins -= excess_length * left_interval_coins
                    # Adjust the interval at the left
                    intervals[left] = (
                        intervals[left][0] + excess_length,
                        intervals[left][1],
                        left_interval_length - excess_length,
                        left_interval_coins
                    )
                    break  # Need to move on to check for max_total_coins
                    
            if curr_total_length == k:
                max_total_coins = max(max_total_coins, curr_total_coins)
        
        return max_total_coins