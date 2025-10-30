# YOUR CODE HERE
import sys

def max_consecutive_free_days(n, d, schedules):
    # Initialize a list to keep track of the number of people free on each day
    free_days = [0] * d
    
    # Count the number of people free on each day
    for schedule in schedules:
        for i in range(d):
            if schedule[i] == 'o':
                free_days[i] += 1
    
    # Initialize variables to track the maximum number of consecutive free days
    max_consecutive = 0
    current_consecutive = 0
    
    # Iterate through the free_days list to find the maximum number of consecutive free days
    for i in range(d):
        if free_days[i] == n:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive

# Read input from stdin
input = sys.stdin.read
data = input().split()
n, d = map(int, data[0].split())
schedules = data[1:]

# Solve the problem and print the result
print(max_consecutive_free_days(n, d, schedules))