def solve():
    N, D = map(int, input().split())
    schedules = []
    for _ in range(N):
        schedules.append(input())

    max_streak = 0
    current_streak = 0
    
    # Iterate through each day (column index j from 0 to D-1)
    for j in range(D):
        all_people_free_this_day = True
        # Check if all N people are free on day j
        for i in range(N): # Iterate through each person (row index i from 0 to N-1)
            if schedules[i][j] == 'x':
                all_people_free_this_day = False
                break  # If one person is not free, no need to check others for this day
        
        if all_people_free_this_day:
            current_streak += 1
        else:
            # This day breaks the current streak of all-free days.
            # Update the maximum streak found so far.
            max_streak = max(max_streak, current_streak)
            # Reset the current streak.
            current_streak = 0
            
    # After the loop, account for a streak that might extend to the last day.
    # This also handles the case where all days were free days (current_streak would hold the total).
    max_streak = max(max_streak, current_streak)
    
    # Print the result
    print(max_streak)

if __name__ == '__main__':
    solve()