class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_durations = {}
        previous_time = None
        
        for event in events:
            index, time = event
            if previous_time is None:
                duration = time
            else:
                duration = time - previous_time
            if index in max_durations:
                if duration > max_durations[index]:
                    max_durations[index] = duration
            else:
                max_durations[index] = duration
            previous_time = time
        
        max_val = float('-inf')
        result_index = None
        
        for idx, val in max_durations.items():
            if val > max_val or (val == max_val and idx < result_index):
                max_val = val
                result_index = idx
        
        return result_index