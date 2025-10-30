class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        button_times = {}
        max_time = 0
        result_index = -1

        for i in range(len(events)):
            index = events[i][0]
            time = events[i][1]

            if index not in button_times:
                button_times[index] = time
            else:
                current_time = time - button_times[index]
                button_times[index] = time
                if current_time > max_time:
                    max_time = current_time
                    result_index = index
                elif current_time == max_time:
                    result_index = min(result_index, index)

        if not button_times:
            return -1 #Handle empty input

        if result_index == -1: #Handle case where only one event per button
            max_time = 0
            result_index = -1
            for index, time in button_times.items():
                if time > max_time:
                    max_time = time
                    result_index = index
                elif time == max_time:
                    result_index = min(result_index, index)

        return result_index