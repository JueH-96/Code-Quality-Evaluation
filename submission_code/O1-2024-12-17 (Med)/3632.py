class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = events[0][1]
        max_button = events[0][0]
        
        for i in range(1, len(events)):
            current_time = events[i][1] - events[i-1][1]  # time to press current button
            current_button = events[i][0]
            
            if current_time > max_time:
                max_time = current_time
                max_button = current_button
            elif current_time == max_time and current_button < max_button:
                # if there is a tie, choose smaller index
                max_button = current_button
        
        return max_button