import sys
input = sys.stdin.read

def max_consecutive_free_days(N, D, schedules):
    # Initialize a list to keep track of free days for all people
    free_days = [True] * D

    # Update the free_days list based on each person's schedule
    for schedule in schedules:
        for day, status in enumerate(schedule):
            if status == 'x':
                free_days[day] = False

    # Find the maximum number of consecutive free days
    max_consecutive = 0
    current_consecutive = 0

    for day in free_days:
        if day:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return max_consecutive

# Read input
data = input().split()
N = int(data[0])
D = int(data[1])
schedules = data[2:]

# Calculate and print the result
result = max_consecutive_free_days(N, D, schedules)
print(result)