def find_max_consecutive_free_days():
    # Read input
    N, D = map(int, input().split())
    schedules = [input().strip() for _ in range(N)]
    
    # For each possible start day, try to find longest consecutive sequence
    max_days = 0
    
    for start in range(D):
        # For each possible length starting from start
        current_length = 0
        for length in range(D - start):
            # Check if all people are free for this day
            all_free = True
            for person in range(N):
                if schedules[person][start + length] == 'x':
                    all_free = False
                    break
            
            if all_free:
                current_length += 1
                max_days = max(max_days, current_length)
            else:
                break
                
    print(max_days)

find_max_consecutive_free_days()