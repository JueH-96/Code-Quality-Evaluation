class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = events[0][1]
        max_index = events[0][0]
        for i in range(1, len(events)):
            time_diff = events[i][1] - events[i - 1][1]
            if time_diff > max_time:
                max_time = time_diff
                max_index = events[i][0]
            elif time_diff == max_time:
                max_index = min(max_index, events[i][0])
        return max_index