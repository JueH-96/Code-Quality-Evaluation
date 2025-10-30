import sys

def solve():
    """
    Reads the calendar parameters and a specific date, then calculates and prints the following day.
    """
    # Read the number of months in a year (M) and days in a month (D) from standard input.
    # The input values are space-separated integers on a single line.
    try:
        M, D = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # This handles potential read errors, though problem constraints suggest valid input.
        return

    # Read the current date: year (y), month (m), and day (d).
    try:
        y, m, d = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        return

    # --- Core Logic to Calculate the Next Day ---

    # 1. Increment the day by one.
    d += 1

    # 2. Check for day rollover.
    # If the new day is greater than the maximum days in a month (D),
    # it means we have moved to the next month.
    if d > D:
        # Reset the day to the 1st.
        d = 1
        # Increment the month.
        m += 1

        # 3. Check for month rollover.
        # After incrementing the month, if it's greater than the maximum
        # months in a year (M), it means we have moved to the next year.
        if m > M:
            # Reset the month to the 1st.
            m = 1
            # Increment the year.
            y += 1

    # Print the resulting date (y', m', d') to standard output.
    # The print function, when given multiple arguments, separates them with spaces by default,
    # which matches the required output format.
    print(y, m, d)

# Execute the main logic.
solve()