def max_consecutive_free_days(N, D, schedules):
    """
    Calculate the maximum number of consecutive days when all people are free.

    Args:
    N (int): The number of people.
    D (int): The number of days.
    schedules (list): A list of strings representing the schedules of each person.

    Returns:
    int: The maximum number of consecutive days when all people are free.
    """
    # Initialize a list to store the combined schedule
    combined_schedule = ['o'] * D

    # Combine the schedules of all people
    for schedule in schedules:
        for i in range(D):
            if schedule[i] == 'x':
                combined_schedule[i] = 'x'

    # Initialize variables to store the maximum and current consecutive free days
    max_free_days = 0
    current_free_days = 0

    # Iterate over the combined schedule
    for day in combined_schedule:
        # If the day is free, increment the current consecutive free days
        if day == 'o':
            current_free_days += 1
            # Update the maximum consecutive free days if necessary
            max_free_days = max(max_free_days, current_free_days)
        # If the day is occupied, reset the current consecutive free days
        else:
            current_free_days = 0

    return max_free_days


# Read the input from stdin
N, D = map(int, input().split())
schedules = [input() for _ in range(N)]

# Calculate and print the maximum consecutive free days
print(max_consecutive_free_days(N, D, schedules))