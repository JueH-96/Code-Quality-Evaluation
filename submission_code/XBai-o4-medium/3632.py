class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        if len(events) == 1:
            return events[0][0]
        
        max_time = events[0][1]
        max_index = events[0][0]
        
        for i in range(1, len(events)):
            current_time = events[i][1] - events[i-1][1]
            if current_time > max_time:
                max_time = current_time
                max_index = events[i][0]
            elif current_time == max_time:
                if events[i][0] < max_index:
                    max_index = events[i][0]
        
        return max_index