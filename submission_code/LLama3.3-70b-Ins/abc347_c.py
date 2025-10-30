import sys

def solve(N, A, B, D):
    """
    Determine if it is possible for all of Takahashi's N plans to be scheduled on holidays.

    Args:
    N (int): The number of plans.
    A (int): The number of holidays in a week.
    B (int): The number of weekdays in a week.
    D (list): A list of days when each plan is scheduled.

    Returns:
    bool: True if it is possible for all plans to be scheduled on holidays, False otherwise.
    """
    # Calculate the total number of days in a week
    total_days = A + B

    # Initialize a set to store the days when plans are scheduled
    scheduled_days = set()

    # Iterate over each plan
    for d in D:
        # Calculate the day of the week when the plan is scheduled
        day = d % total_days

        # If the day is a weekday, return False
        if day >= A:
            return False

        # Add the day to the set of scheduled days
        scheduled_days.add(day)

    # If all plans can be scheduled on holidays, return True
    return True

def main():
    # Read input from stdin
    N, A, B = map(int, sys.stdin.readline().split())
    D = list(map(int, sys.stdin.readline().split()))

    # Solve the problem
    result = solve(N, A, B, D)

    # Print the result to stdout
    if result:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()