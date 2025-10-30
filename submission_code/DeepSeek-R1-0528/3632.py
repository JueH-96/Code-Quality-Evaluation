class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        ans = events[0][0]
        max_time = events[0][1]
        n = len(events)
        for i in range(1, n):
            current_time = events[i][1] - events[i-1][1]
            if current_time > max_time:
                max_time = current_time
                ans = events[i][0]
            elif current_time == max_time:
                if events[i][0] < ans:
                    ans = events[i][0]
        return ans