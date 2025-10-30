import collections
from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        """
        Calculates the maximum amount of coins obtainable by collecting k consecutive bags.

        This solution uses a sweep-line algorithm. Let C(x) be the total coins in the
        window starting at position x, i.e., [x, x + k - 1]. We want to find max(C(x)).

        The rate of change of C(x) as x increases is given by f(x) = C(x+1) - C(x) =
        coins(x+k) - coins(x). This function f(x) is piecewise constant, changing
        only at "event points" where x or x+k crosses a segment boundary.

        The algorithm identifies these event points, calculates the sum for the first
        event point's window, and then sweeps across the sorted event points, updating
        the sum based on the constant rate of change between them. The maximum sum
        encountered during the sweep is the answer.
        """
        
        # A map to store event points and the corresponding change in the derivative f(x).
        # key: position, value: change in f(x)
        events = collections.defaultdict(int)
        for l, r, c in coins:
            # Contribution from coins(t+k):
            # Event when window start `x` makes `x+k` enter a segment [l,r]
            events[l - k] += c
            # Event when window start `x` makes `x+k` leave a segment [l,r]
            events[r + 1 - k] -= c
            
            # Contribution from -coins(t):
            # Event when window start `x` enters a segment [l,r]
            events[l] -= c
            # Event when window start `x` leaves a segment [l,r]
            events[r + 1] += c

        if not events:
            return 0

        # Get the sorted list of unique event positions
        sorted_pos = sorted(events.keys())
        
        # Calculate the initial sum for the window starting at the first event position.
        # This is our base case for the sweep-line.
        start_pos = sorted_pos[0]
        window_start, window_end = start_pos, start_pos + k - 1
        
        current_sum = 0
        for l, r, c in coins:
            # Calculate overlap between coin segment [l, r] and the window
            overlap_start = max(l, window_start)
            overlap_end = min(r, window_end)
            
            if overlap_start <= overlap_end:
                overlap_len = overlap_end - overlap_start + 1
                current_sum += overlap_len * c
        
        max_sum = current_sum
        
        # f_val is the rate of change of the window sum (the derivative).
        # It's constant between any two consecutive event points.
        f_val = 0
        
        # Sweep through the event points to find the maximum window sum.
        for i in range(len(sorted_pos) - 1):
            pos = sorted_pos[i]
            next_pos = sorted_pos[i+1]
            
            # Update f_val for the interval [pos, next_pos - 1] using the change at `pos`.
            f_val += events[pos]
            
            # The window sum changes linearly between `pos` and `next_pos`.
            # Update the sum from C(pos) to C(next_pos).
            interval_len = next_pos - pos
            current_sum += interval_len * f_val
            
            # The maximum sum can occur at any event point.
            max_sum = max(max_sum, current_sum)
            
        return max_sum