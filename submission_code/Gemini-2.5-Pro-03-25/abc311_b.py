import sys

def solve():
    """
    Reads the input, determines the days when all people are free,
    finds the maximum number of consecutive such days, and prints the result.
    """
    # Read N (number of people) and D (number of days) from the first line of input
    n, d = map(int, sys.stdin.readline().split())

    # Initialize a boolean list 'all_free' of size D.
    # This list will track whether all people are free on each day.
    # Initially, assume all days are potentially free for everyone.
    all_free = [True] * d

    # Process the schedule for each of the N people
    for _ in range(n):
        # Read the schedule string (length D) for the current person
        schedule = sys.stdin.readline().strip()

        # Iterate through each day index from 0 to D-1
        for day_index in range(d):
            # Optimization: If we already determined that not everyone is free
            # on this day based on previous people's schedules, we can skip checking
            # the current person's schedule for this day.
            if not all_free[day_index]:
                continue

            # Check the current person's status on this day.
            # If the person is occupied ('x'), then not everyone is free on this day.
            if schedule[day_index] == 'x':
                # Mark this day as not free for everyone.
                all_free[day_index] = False
                # We still need to continue checking the rest of the days for this person.
                # The `if not all_free[day_index]: continue` check handles efficiency
                # for subsequent people on this now-determined non-free day.

    # Now, calculate the maximum number of consecutive days when everyone is free.
    max_consecutive_days = 0
    current_consecutive_days = 0

    # Iterate through the 'all_free' status list for each day
    for is_free in all_free:
        if is_free:
            # If the day is marked as free for everyone, increment the counter
            # for the current consecutive streak.
            current_consecutive_days += 1
            # Update the overall maximum consecutive streak found so far.
            max_consecutive_days = max(max_consecutive_days, current_consecutive_days)
        else:
            # If the day is not free for everyone, the consecutive streak is broken.
            # Reset the counter for the current streak to 0.
            current_consecutive_days = 0

    # Print the final result, which is the maximum length of consecutive days
    # where all N people were free. If no such days exist, max_consecutive_days will be 0.
    print(max_consecutive_days)

# Call the main function to run the solution
solve()