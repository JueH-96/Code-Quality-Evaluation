class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        if not events:
            return -1
        
        max_time = -1
        result_button = -1
        prev_time = events[0][1]
        
        for i in range(1, len(events)):
            current_time = events[i][1]
            time_diff = current_time - prev_time
            if time_diff > max_time:
                max_time = time_diff
                result_button = events[i][0]
            elif time_diff == max_time:
                if events[i][0] < result_button:
                    result_button = events[i][0]
            prev_time = current_time
        
        # Handle the first event
        if events[0][1] > max_time:
            max_time = events[0][1]
            result_button = events[0][0]
        
        return result_button