class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = events[0][1]  # Time for the first button
        result_index = events[0][0]  # Index of the first button
        
        for i in range(1, len(events)):
            time_taken = events[i][1] - events[i-1][1]
            
            if time_taken > max_time:
                max_time = time_taken
                result_index = events[i][0]
            elif time_taken == max_time and events[i][0] < result_index:
                result_index = events[i][0]
        
        return result_index