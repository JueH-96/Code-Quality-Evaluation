class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_time = events[0][1]
        ans = events[0][0]
        
        for i in range(1, len(events)):
            current_time = events[i][1] - events[i - 1][1]
            if current_time > max_time or (current_time == max_time and events[i][0] < ans):
                max_time = current_time
                ans = events[i][0]
        
        return ans