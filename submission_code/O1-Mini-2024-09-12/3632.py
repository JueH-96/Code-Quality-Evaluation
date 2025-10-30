class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = -1
        result_button = None
        previous_time = 0

        for i, (index_i, time_i) in enumerate(events):
            if i == 0:
                time_taken = time_i
            else:
                time_taken = time_i - previous_time

            if time_taken > max_time or (time_taken == max_time and index_i < result_button):
                max_time = time_taken
                result_button = index_i

            previous_time = time_i

        return result_button