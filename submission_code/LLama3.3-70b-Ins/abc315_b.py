import sys

def find_middle_day():
    # Read the number of months
    M = int(input())

    # Read the number of days in each month
    D = list(map(int, input().split()))

    # Calculate the total number of days in the year
    total_days = sum(D)

    # Calculate the middle day
    middle_day = (total_days + 1) // 2

    # Initialize the current month and day
    month = 1
    day = 1

    # Iterate over the months
    for i in range(M):
        # If the middle day is within the current month
        if middle_day <= D[i]:
            # Update the day
            day = middle_day
            # Break the loop
            break
        else:
            # Subtract the days of the current month from the middle day
            middle_day -= D[i]
            # Move to the next month
            month += 1

    # Print the result
    print(month, day)

# Call the function
find_middle_day()