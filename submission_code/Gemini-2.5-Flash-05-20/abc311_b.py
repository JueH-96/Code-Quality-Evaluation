# YOUR CODE HERE
def solve():
    N, D = map(int, input().split())
    
    schedules = []
    for _ in range(N):
        schedules.append(input())
        
    max_consecutive_days = 0
    current_consecutive_days = 0
    
    for day_idx in range(D):
        all_free_today = True
        for person_idx in range(N):
            if schedules[person_idx][day_idx] == 'x':
                all_free_today = False
                break # If one person is occupied, not everyone is free
        
        if all_free_today:
            current_consecutive_days += 1
        else:
            # The streak of free days is broken.
            # Update max_consecutive_days with the current streak's length.
            max_consecutive_days = max(max_consecutive_days, current_consecutive_days)
            # Reset current_consecutive_days for the next potential streak.
            current_consecutive_days = 0
            
    # After the loop, one last check is needed for a streak that might end on the last day
    max_consecutive_days = max(max_consecutive_days, current_consecutive_days)
    
    print(max_consecutive_days)

solve()