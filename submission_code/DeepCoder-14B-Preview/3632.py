class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        prev_time = 0
        max_duration = 0
        result_index = -1
        for event in events:
            current_time = event[1]
            duration = current_time - prev_time
            if duration > max_duration:
                max_duration = duration
                result_index = event[0]
            elif duration == max_duration:
                if event[0] < result_index:
                    result_index = event[0]
            prev_time = current_time
        return result_index