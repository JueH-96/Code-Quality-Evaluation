import sys

def solve():
    """
    Reads input, simulates the process, and prints the result.
    """
    try:
        # Read the number of presses (N) and the cooldown period (C).
        # sys.stdin.readline is used for efficiency in competitive programming.
        n_str, c_str = sys.stdin.readline().split()
        n = int(n_str)
        c = int(c_str)

        # Read the list of press times. They are guaranteed to be sorted.
        times = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Graceful exit on malformed or empty input, though not
        # expected under problem constraints.
        return

    # Initialize a counter for the candies received.
    candy_count = 0

    # Initialize the time of the last candy reception.
    # Using negative infinity ensures the condition for the first candy is always met.
    last_candy_time = float('-inf')

    # Iterate through each press time.
    for current_press_time in times:
        # Check if at least C seconds have passed since the last candy was received.
        if current_press_time - last_candy_time >= c:
            # If so, Takahashi receives a candy.
            candy_count += 1
            # Update the time of the last candy receipt to the current time.
            last_candy_time = current_press_time
    
    # Print the total number of candies received to standard output.
    print(candy_count)

solve()