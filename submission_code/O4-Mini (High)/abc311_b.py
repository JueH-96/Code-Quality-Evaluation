def main():
    import sys
    input = sys.stdin.readline
    
    # Read N (number of people) and D (number of days)
    N, D = map(int, input().split())
    
    # Read the schedules for each person
    schedules = [input().strip() for _ in range(N)]
    
    max_consecutive = 0
    current_streak = 0
    
    # Iterate over each day
    for day in range(D):
        # Check if all people are free on this day
        all_free = True
        for i in range(N):
            if schedules[i][day] == 'x':
                all_free = False
                break
        
        if all_free:
            # Extend the current streak of free days
            current_streak += 1
        else:
            # Day is not free for all; finalize the current streak
            if current_streak > max_consecutive:
                max_consecutive = current_streak
            current_streak = 0
    
    # In case the longest streak goes until the last day
    if current_streak > max_consecutive:
        max_consecutive = current_streak
    
    # Print the result (0 if no fully-free days exist)
    print(max_consecutive)

if __name__ == "__main__":
    main()