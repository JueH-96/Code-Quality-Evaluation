class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        prev_times = {}
        max_time = 0
        result = None
        for event in events:
            index = event[0]
            current_time = event[1]
            if index not in prev_times:
                duration = current_time
                prev_times[index] = current_time
            else:
                duration = current_time - prev_times[index]
                prev_times[index] = current_time
            if duration > max_time:
                max_time = duration
                result = index
            elif duration == max_time:
                if index < result:
                    result = index
        return result