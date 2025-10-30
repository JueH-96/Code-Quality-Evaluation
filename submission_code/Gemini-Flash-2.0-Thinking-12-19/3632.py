class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = -1
        result_index = -1
        previous_time = 0
        for index, time in events:
            if previous_time == 0:
                current_time_taken = time
            else:
                current_time_taken = time - previous_time

            if current_time_taken > max_time:
                max_time = current_time_taken
                result_index = index
            elif current_time_taken == max_time:
                if result_index == -1 or index < result_index:
                    result_index = index

            previous_time = time
        return result_index