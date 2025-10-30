def count_repdigits(N, D):
    """
    Count the number of days in a year of AtCoder Kingdom that have "repdigits" dates.

    Args:
    N (int): The number of months in a year.
    D (list): A list of the number of days in each month.

    Returns:
    int: The number of days with repdigit dates.
    """
    count = 0
    for i in range(1, N + 1):
        # Convert the month number to a string to easily access its digits
        month_str = str(i)
        # Get the first digit of the month number
        digit = month_str[0]
        # Check each day in the month
        for j in range(1, D[i - 1] + 1):
            # Convert the day number to a string to easily access its digits
            day_str = str(j)
            # Check if all digits in the day number are the same as the first digit of the month number
            if all(d == digit for d in day_str):
                count += 1
    return count

# Read the input from stdin
N = int(input())
D = list(map(int, input().split()))

# Count the number of days with repdigit dates
count = count_repdigits(N, D)

# Write the answer to stdout
print(count)