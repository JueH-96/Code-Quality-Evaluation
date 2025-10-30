class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time_taken = {}
        last_press_time = {}

        for index, time in events:
            if index not in last_press_time:
                time_taken = time
            else:
                time_taken = time - last_press_time[index]

            max_time_taken[index] = max(max_time_taken.get(index, 0), time_taken)
            last_press_time[index] = time

        if not max_time_taken:
            return -1  # Should not happen given constraints

        max_time = -1
        for time in max_time_taken.values():
            max_time = max(max_time, time)

        result_button = float('inf')
        for button, time in max_time_taken.items():
            if time == max_time:
                result_button = min(result_button, button)

        return result_button