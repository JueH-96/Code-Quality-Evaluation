class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = 0
        button_index = 0
        prev_time = events[0][1]
        
        for event in events[1:]:
            index, time = event
            duration = time - prev_time
            if duration > max_time:
                max_time = duration
                button_index = index
            prev_time = time
        
        return button_index