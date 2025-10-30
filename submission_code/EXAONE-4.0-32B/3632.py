class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        max_duration = events[0][1]
        candidate_index = events[0][0]
        
        for i in range(1, len(events)):
            duration = events[i][1] - events[i-1][1]
            if duration > max_duration:
                max_duration = duration
                candidate_index = events[i][0]
            elif duration == max_duration:
                if events[i][0] < candidate_index:
                    candidate_index = events[i][0]
                    
        return candidate_index