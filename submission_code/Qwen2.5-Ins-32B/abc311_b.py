import sys

def max_consecutive_free_days(N, D, schedules):
    # Initialize a list to keep track of free days for each day
    free_days = [0] * D
    
    # Count the number of free people for each day
    for schedule in schedules:
        for j in range(D):
            if schedule[j] == 'o':
                free_days[j] += 1
    
    # Initialize variables to track the maximum number of consecutive free days
    max_free_days = 0
    current_free_days = 0
    
    # Iterate through the free_days list to find the maximum number of consecutive free days
    for i in range(D):
        if free_days[i] == N:
            current_free_days += 1
            max_free_days = max(max_free_days, current_free_days)
        else:
            current_free_days = 0
    
    return max_free_days

# Read input from stdin
input_data = sys.stdin.read().splitlines()
N, D = map(int, input_data[0].split())
schedules = input_data[1:]

# Solve the problem and write the answer to stdout
print(max_consecutive_free_days(N, D, schedules))