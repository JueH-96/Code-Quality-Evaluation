class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = 0
        res = 0
        prev_time = 0
        for index, time in events:
            curr_time = time - prev_time
            if curr_time > max_time:
                max_time = curr_time
                res = index
            elif curr_time == max_time:
                res = min(res, index)
            prev_time = time
        return res