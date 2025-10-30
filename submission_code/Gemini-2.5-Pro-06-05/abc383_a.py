# YOUR CODE HERE
import sys

def solve():
    """
    Solves the humidifier problem by simulating the water level over time.
    """
    try:
        # Read N, the number of water additions.
        n_str = sys.stdin.readline()
        if not n_str:
            return
        n = int(n_str)
    except (ValueError, IndexError):
        # Gracefully handle empty or malformed input for N.
        return

    # Initialize the state of the humidifier.
    # `current_water` tracks the amount of water in liters.
    # `last_time` tracks the time of the last recorded event.
    current_water = 0
    last_time = 0

    # Process each of the N events in chronological order.
    for _ in range(n):
        try:
            line = sys.stdin.readline()
            if not line:
                break
            # Read the time (T) and volume (V) for the current event.
            t, v = map(int, line.split())
        except (ValueError, IndexError):
            # Skip any malformed event lines.
            continue

        # Calculate the time elapsed since the last event.
        time_elapsed = t - last_time

        # Water leaks at 1 liter per unit time. The water level decreases by
        # `time_elapsed`, but cannot drop below zero.
        current_water = max(0, current_water - time_elapsed)

        # Add the new water for the current event.
        current_water += v

        # Update the time of the last event to the current event's time.
        last_time = t

    # Print the final amount of water after the last addition.
    print(current_water)

# It's good practice to wrap the execution in a main block.
if __name__ == "__main__":
    solve()