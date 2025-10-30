class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        if not events:
            return -1  # Or handle empty case as needed

        max_time = 0
        max_index = events[0][0]
        
        last_press_time = {}

        for index, time in events:
            if index not in last_press_time:
                time_diff = time
            else:
                time_diff = time - last_press_time[index]
            
            if time_diff > max_time:
                max_time = time_diff
                max_index = index
            elif time_diff == max_time and index < max_index:
                max_index = index
            
            last_press_time[index] = time
        
        return max_index