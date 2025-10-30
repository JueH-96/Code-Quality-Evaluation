# YOUR CODE HERE
import sys

def solve():
    """
    Reads Takahashi's height, simulates the plant's growth, and prints
    the first day the plant is taller.
    """
    try:
        H = int(sys.stdin.readline())
    except (IOError, ValueError):
        # Gracefully handle cases with no input or invalid input,
        # although problem constraints guarantee valid input.
        return

    # Initialize the plant's height and the day counter.
    plant_height = 0
    day = 0
    
    # `growth_per_night` will hold the value of 2^i for the night of day i.
    # It starts at 1 (for 2^0).
    growth_per_night = 1

    # The loop continues as long as the plant's height is not strictly
    # greater than Takahashi's height.
    while plant_height <= H:
        # The plant grows during the night.
        plant_height += growth_per_night
        
        # For the next night, the growth amount doubles.
        growth_per_night *= 2
        
        # A day passes.
        day += 1

    # The loop terminates when plant_height > H. The value of 'day'
    # at this point represents the first day on which this condition is met.
    print(day)

# Run the solution
solve()