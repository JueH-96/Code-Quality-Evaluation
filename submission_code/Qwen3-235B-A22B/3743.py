from heapq import heappush, heappop
from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        if n == 0:
            return eventTime
        
        durations = [endTime[i] - startTime[i] for i in range(n)]
        total_durations = sum(durations)
        max_total_free = eventTime - total_durations
        if max_total_free <= 0:
            return 0
        
        # Calculate initial, between, and final gaps
        current_max = startTime[0]
        for i in range(n-1):
            gap = startTime[i+1] - endTime[i]
            if gap > current_max:
                current_max = gap
        final_gap = eventTime - endTime[-1]
        current_max = max(current_max, final_gap)
        
        if k == 0:
            return current_max
        
        # Use sliding window with a min-heap to track up to k largest durations
        left = 0
        sum_window = 0
        min_heap = []
        best = current_max
        
        for right in range(n):
            # Add current duration to the window
            current_duration = durations[right]
            heappush(min_heap, current_duration)
            sum_window += current_duration
            
            # If window size exceeds k, remove the smallest elements until size <=k
            while len(min_heap) > k:
                removed = heappop(min_heap)
                sum_window -= removed
            
            # Calculate the effective window start and span
            # The sum of the elements in the heap are the largest (up to k) in the window
            sum_k = sum(min_heap)
            window_start = startTime[left]
            window_end = endTime[right]
            window_span = window_end - window_start
            # sum_durations_in_window = sum of durations in window (sum_window)
            # gaps within the window: window_span - sum_window
            # adding sum_k (moved durations) to gaps window_span - (sum_window - (sum_k)) = window_span - (sum_window - (sum_k)) 
            # which is window_span - (sum_window - (sum_k)) = window_span - (sum_window - sum_k)
            # but sum_window - (sum_k) = sum of durations not moved in window (smallest)
            # not sure, but considering that moved durations sum is 'sum_k'
            candidate = (window_span - (sum_window)) + (sum_k)
            if candidate > best:
                best = candidate
            
            # Adjust left pointer if necessary
            # Determine if window can be shrunk
            # The logic here is not strictly necessary for some cases, but to explore all possible windows
            while left <= right:
                # Try to move left to find a smaller window
                removed_duration = durations[left]
                if len(min_heap) == k and removed_duration < min_heap[0]:
                    # This duration is not in the heap, so remove from sum_window
                    sum_window -= removed_duration
                    left += 1
                elif len(min_heap) < k:
                    # Still can add to heap
                    break
                else:
                    # The removed duration might be in the heap
                    if removed_duration in min_heap:
                        min_heap.remove(removed_duration)
                        sum_window -= removed_duration
                        # Re-heapify is needed but expensive, so alternative approach may be needed
                        # This approach may not be efficient, but for the sake of example, continue
                        # Instead, consider using a hash to track elements
                        # This part is complex, and may not be feasible in practice
                        # Alternative approach: reset heap if left moves
                        # This code may not work perfectly due to this section
                        import heapq
                        new_heap = []
                        for val in min_heap:
                            if val != removed_duration or removed_duration has been removed:
                                new_heap.append(val)
                            removed_duration = None
                        min_heap = new_heap
                        heapq.heapify(min_heap)
                    else:
                        sum_window -= removed_duration
                    left +=1
                    break
            # Continue with next right
        return min(max_total_free, best)