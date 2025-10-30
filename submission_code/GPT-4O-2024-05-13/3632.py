class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = events[0][1]
        button_index = events[0][0]
        
        for i in range(1, len(events)):
            time_taken = events[i][1] - events[i-1][1]
            if time_taken > max_time or (time_taken == max_time and events[i-1][0] < button_index):
                max_time = time_taken
                button_index = events[i-1][0]
        
        return button_index