import sys

def solve():
    """
    Reads snake data and for each day k from 1 to D, calculates and prints
    the weight of the heaviest snake when every snake's length has increased by k.
    """
    try:
        # Read the number of snakes (N) and the number of days (D).
        first_line = sys.stdin.readline()
        if not first_line:
            return  # Handle empty input
        N, D = map(int, first_line.split())

        # Read the thickness (T) and initial length (L) for each of the N snakes
        # and store them in a list of tuples.
        snakes_data = []
        for _ in range(N):
            line = sys.stdin.readline()
            if not line:
                break # Handle premature end of input
            T, L = map(int, line.split())
            snakes_data.append((T, L))
    except (ValueError, IndexError):
        # In case of malformed input, exit gracefully.
        return
        
    # If no snake data was provided, there's nothing to process.
    if not snakes_data:
        return

    # For each day k, from 1 to D...
    for k in range(1, D + 1):
        # Calculate the weight of each snake for the current day k.
        # The new length is (L_i + k), so the weight is T_i * (L_i + k).
        
        # We use a generator expression `(T * (L + k) for T, L in snakes_data)`
        # which is an efficient way to compute the weight for each snake on the fly.
        # The `max()` function then finds the maximum value from the generated weights.
        max_weight_for_day = max(T * (L + k) for T, L in snakes_data)
        
        # Print the maximum weight for the current day.
        print(max_weight_for_day)

# It's standard practice in competitive programming to have the code run directly.
solve()