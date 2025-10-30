class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        # For first event, time taken is just the time at which it was pressed
        max_time = events[0][1]
        max_index = events[0][0]
        
        # For subsequent events, time taken is difference between consecutive presses
        for i in range(1, len(events)):
            current_time = events[i][1] - events[i-1][1]
            
            # Update max_time and max_index if current_time is larger
            # or if times are equal but current index is smaller
            if current_time > max_time or (current_time == max_time and events[i][0] < max_index):
                max_time = current_time
                max_index = events[i][0]
                
        return max_index