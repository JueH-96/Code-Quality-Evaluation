def max_consecutive_free_days(N, D, schedules):
    # Create an array to track the availability of all people for each day
    availability = [1] * D  # Start with all days available (1 means free, 0 means occupied)

    # Check each person's schedule
    for schedule in schedules:
        for j in range(D):
            if schedule[j] == 'x':
                availability[j] = 0  # Mark the day as occupied if any person is occupied

    # Now find the maximum length of consecutive free days
    max_length = 0
    current_length = 0

    for day in availability:
        if day == 1:  # If the day is free
            current_length += 1
        else:  # If the day is occupied
            max_length = max(max_length, current_length)
            current_length = 0

    # Final check in case the longest sequence ends at the last day
    max_length = max(max_length, current_length)

    return max_length

import sys
input = sys.stdin.read
data = input().splitlines()

N, D = map(int, data[0].split())
schedules = data[1:N+1]

result = max_consecutive_free_days(N, D, schedules)
print(result)