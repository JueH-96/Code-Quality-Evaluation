class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        meetings = sorted(zip(startTime, endTime), key=lambda x: x[0])
        
        # Initialize the maximum free time and the current free time
        max_free_time = 0
        current_free_time = 0
        
        # Iterate through the meetings
        for i in range(n):
            # Calculate the current free time
            if i == 0:
                current_free_time = meetings[0][0]
            else:
                current_free_time = meetings[i][0] - meetings[i-1][1]
            
            # Update the maximum free time
            max_free_time = max(max_free_time, current_free_time)
            
            # If we have k rescheduling opportunities left
            if k > 0:
                # Try to reschedule the current meeting
                new_start = meetings[i][0] + 1
                new_end = meetings[i][1] + 1
                
                # Check if the rescheduled meeting is within the event time
                if new_end <= eventTime:
                    # Calculate the new free time
                    if i == 0:
                        new_free_time = new_start
                    else:
                        new_free_time = new_start - meetings[i-1][1]
                    
                    # Update the maximum free time
                    max_free_time = max(max_free_time, new_free_time)
                    
                    # Decrement the number of rescheduling opportunities
                    k -= 1
        
        # If no free time is possible, return -1
        return max_free_time if max_free_time > 0 else -1