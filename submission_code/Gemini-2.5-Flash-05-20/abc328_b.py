import sys

def get_repdigit_char(num):
    """
    Checks if a number is a repdigit and returns the repeated digit character.
    A number is a repdigit if all its digits are the same (e.g., 1, 7, 11, 55).
    Returns None if the number is not a repdigit.
    """
    s = str(num)
    
    # An empty string is not a valid number for this problem's context (num >= 1)
    if not s:
        return None
    
    first_char = s[0]
    
    # Check if all characters in the string are the same as the first one
    # This ensures it's a repdigit (e.g., "11" is repdigit '1', "23" is not)
    if all(c == first_char for c in s):
        return first_char
    else:
        return None

def solve():
    """
    Reads the input, calculates the number of repdigit dates, and prints the result.
    A date (month i, day j) is a repdigit date if:
    1. Month 'i' is a repdigit number.
    2. Day 'j' is a repdigit number.
    3. The common digit for 'i' is the same as the common digit for 'j'.
    """
    
    # Read N (number of months)
    N = int(sys.stdin.readline())
    
    # Read D (list of days in each month)
    D = list(map(int, sys.stdin.readline().split()))

    count = 0 # Initialize counter for repdigit dates

    # Iterate through each month. Month numbers are 1-indexed.
    # The D list is 0-indexed, so month_idx goes from 0 to N-1.
    for month_idx in range(N):
        i = month_idx + 1 # Current month number (1 to N)
        max_days_in_month = D[month_idx] # Maximum days for the current month

        # Determine if the current month number 'i' is a repdigit and get its repeated digit character.
        # Example: if i=1, repdigit_char_i = '1'. If i=11, repdigit_char_i = '1'. If i=10, repdigit_char_i = None.
        repdigit_char_i = get_repdigit_char(i)

        # If the month 'i' itself is not a repdigit, then no date in this month
        # can satisfy the repdigit date condition, so we can skip to the next month.
        if repdigit_char_i is None:
            continue

        # If month 'i' is a repdigit, iterate through each day in this month.
        # Day numbers are 1-indexed, so j goes from 1 to max_days_in_month.
        for j in range(1, max_days_in_month + 1):
            # Determine if the current day number 'j' is a repdigit and get its repeated digit character.
            repdigit_char_j = get_repdigit_char(j)

            # Check if day 'j' is a repdigit AND its repeated digit character is the same
            # as the repeated digit character of month 'i'.
            if repdigit_char_j is not None and repdigit_char_i == repdigit_char_j:
                count += 1 # If both conditions are met, increment the counter
    
    # Print the total count of repdigit dates
    print(count)

# Call the solve function to execute the program
if __name__ == '__main__':
    solve()