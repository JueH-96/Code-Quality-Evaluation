import sys

def solve():
    """
    Reads the calendar specification and counts the number of repdigit dates.
    """
    try:
        # Read N, the number of months.
        n = int(sys.stdin.readline())
        # Read D_list, the number of days in each month.
        d_list = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Gracefully handle cases with no input, though problem constraints guarantee valid input.
        return

    # Initialize a counter for the number of repdigit dates.
    count = 0

    # Iterate through each possible repeating digit (d from 1 to 9).
    for d in range(1, 10):
        # Form the 1-digit and 2-digit repdigit numbers from d.
        num1 = d
        num2 = d * 11

        # Case 1: Check dates in a 1-digit repdigit month (e.g., month 1, 2, ..., 9).
        if num1 <= n:
            # The list of days is 0-indexed, but months are 1-indexed.
            days_in_month = d_list[num1 - 1]
            
            # Check if the 1-digit repdigit day is valid.
            if num1 <= days_in_month:
                count += 1
            
            # Check if the 2-digit repdigit day is valid.
            if num2 <= days_in_month:
                count += 1
        
        # Case 2: Check dates in a 2-digit repdigit month (e.g., month 11, 22, ..., 99).
        if num2 <= n:
            # The list of days is 0-indexed, but months are 1-indexed.
            days_in_month = d_list[num2 - 1]

            # Check if the 1-digit repdigit day is valid.
            if num1 <= days_in_month:
                count += 1
            
            # Check if the 2-digit repdigit day is valid.
            if num2 <= days_in_month:
                count += 1
                
    # Print the final total count to stdout.
    print(count)

solve()