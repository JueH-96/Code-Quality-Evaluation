class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_duration = -1
        ans_index = None
        prev_time = 0

        for i, (index_i, time_i) in enumerate(events):
            if i == 0:
                duration = time_i
            else:
                duration = time_i - events[i-1][1]

            if duration > max_duration:
                max_duration = duration
                ans_index = index_i
            elif duration == max_duration:
                ans_index = min(ans_index, index_i)
        
        return ans_index