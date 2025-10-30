class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = 0
        max_index = events[0][0]
        prev_time = events[0][1]
        
        for i in range(1, len(events)):
            current_index, current_time = events[i]
            time_diff = current_time - prev_time
            
            if time_diff > max_time:
                max_time = time_diff
                max_index = current_index
            elif time_diff == max_time:
                max_index = min(max_index, current_index)
            
            prev_time = current_time
        
        return max_index