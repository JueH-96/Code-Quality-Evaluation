class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = 0
        max_index = events[0][0]
        prev_time = events[0][1]

        for i in range(1, len(events)):
            index, time = events[i]
            duration = time - prev_time
            if duration > max_time or (duration == max_time and index < max_index):
                max_time = duration
                max_index = index
            prev_time = time

        return max_index