def solve():
    """
    This function reads the number of weeks and the daily step counts,
    calculates the total steps for each week, and prints the results.
    """
    # Read the number of weeks, N.
    try:
        N = int(input())
    except (ValueError, EOFError):
        # This handles cases of empty input or non-integer input, though
        # problem constraints suggest valid input.
        return

    # Read the 7*N daily step counts into a list of integers.
    try:
        A = list(map(int, input().split()))
    except (ValueError, EOFError):
        return

    # A list to store the sum of steps for each week, as per the problem's B_i notation.
    B = []

    # Iterate N times, once for each week.
    for i in range(N):
        # The steps for the i-th week (0-indexed) start at index i*7.
        start_index = i * 7
        # The slice goes up to, but does not include, the end_index.
        end_index = start_index + 7
        
        # Extract the steps for the current week using list slicing.
        week_steps = A[start_index:end_index]
        
        # Calculate the sum for the current week and append it to our results list.
        B.append(sum(week_steps))

    # Print the final list of weekly totals.
    # The `*` operator unpacks the list `B`, passing each element as a
    # separate argument to the print function. By default, print separates
    # arguments with a space.
    print(*B)

# Execute the solution.
solve()