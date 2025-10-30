# YOUR CODE HERE
def max_consecutive_free_days(N, D, schedules):
    # Initialize a list to keep track of days when all people are free
    all_free = [True] * D
    
    # Check each day for all people
    for schedule in schedules:
        for day in range(D):
            if schedule[day] == 'x':
                all_free[day] = False
    
    # Find the maximum consecutive free days
    max_consecutive = 0
    current_consecutive = 0
    
    for day in all_free:
        if day:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive

# Read input
N, D = map(int, input().split())
schedules = [input().strip() for _ in range(N)]

# Calculate and print the result
result = max_consecutive_free_days(N, D, schedules)
print(result)