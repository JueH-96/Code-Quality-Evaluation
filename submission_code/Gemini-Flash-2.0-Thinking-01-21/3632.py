class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = 0
        max_button_index = -1
        prev_time = 0
        for i in range(len(events)):
            index, time = events[i]
            current_time_taken = 0
            if i == 0:
                current_time_taken = time
            else:
                current_time_taken = time - prev_time
            if current_time_taken > max_time:
                max_time = current_time_taken
                max_button_index = index
            elif current_time_taken == max_time:
                if max_button_index == -1 or index < max_button_index:
                    max_button_index = index
            prev_time = time
        return max_button_index