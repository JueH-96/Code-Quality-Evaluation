class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        if not events:
            return 0  # According to constraints, this case shouldn't occur
        
        max_time = events[0][1]
        result_button = events[0][0]
        
        for i in range(1, len(events)):
            current_time = events[i][1] - events[i-1][1]
            current_button = events[i][0]
            
            if current_time > max_time:
                max_time = current_time
                result_button = current_button
            elif current_time == max_time:
                if current_button < result_button:
                    result_button = current_button
        
        return result_button