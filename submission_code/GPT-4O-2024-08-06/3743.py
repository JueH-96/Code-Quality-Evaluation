class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        
        # Calculate the initial free times between meetings
        free_times = []
        for i in range(1, n):
            free_times.append(startTime[i] - endTime[i - 1])
        
        # Add free time before the first meeting and after the last meeting
        free_times.insert(0, startTime[0] - 0)
        free_times.append(eventTime - endTime[-1])
        
        # Sort the free times in descending order
        free_times.sort(reverse=True)
        
        # Maximize the free time by considering the largest k free times
        max_free_time = sum(free_times[:k + 1])
        
        return max_free_time