class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = events[0][1]  # First button's time
        result = events[0][0]    # First button's index
        
        for i in range(1, len(events)):
            time_taken = events[i][1] - events[i-1][1]
            
            # Update if we found a longer time, or same time but smaller index
            if time_taken > max_time or (time_taken == max_time and events[i][0] < result):
                max_time = time_taken
                result = events[i][0]
        
        return result