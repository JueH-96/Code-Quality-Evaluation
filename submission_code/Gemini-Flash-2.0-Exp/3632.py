class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        times = {}
        for index, time in events:
            if index not in times:
                times[index] = 0
        
        times[events[0][0]] = events[0][1]
        
        for i in range(1, len(events)):
            times[events[i][0]] = times.get(events[i][0], 0) + (events[i][1] - events[i-1][1])
        
        max_time = 0
        button_index = -1
        
        for index, time in times.items():
            if time > max_time:
                max_time = time
                button_index = index
            elif time == max_time:
                if index < button_index or button_index == -1:
                    button_index = index
        
        return button_index