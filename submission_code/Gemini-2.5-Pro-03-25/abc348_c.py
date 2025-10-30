# YOUR CODE HERE
import sys

# Use input = sys.stdin.readline for faster input reading, especially for large N
input = sys.stdin.readline

def solve():
    # Read the number of bean types from the first line of input
    n = int(input())

    # Dictionary to store the minimum deliciousness found for each distinct color.
    # The keys will be the color identifiers (C_i), and the values will be the
    # minimum deliciousness (A_i) encountered so far for that color.
    min_deliciousness_map = {}

    # Iterate through each of the N bean types
    for _ in range(n):
        # Read the deliciousness (A_i) and color (C_i) for the current bean type.
        # map(int, input().split()) efficiently reads space-separated integers on a line.
        deliciousness, color = map(int, input().split())

        # Check if this color has already been encountered and recorded in the map
        if color in min_deliciousness_map:
            # If the color exists in the map, we compare the current bean's deliciousness
            # with the stored minimum deliciousness for this color. We update the stored
            # value only if the current bean's deliciousness is lower. This ensures
            # that the map always holds the absolute minimum deliciousness for each color seen so far.
            min_deliciousness_map[color] = min(min_deliciousness_map[color], deliciousness)
        else:
            # If this is the first time encountering this color, we add it to the map
            # and record the current bean's deliciousness as the minimum (since it's the only one seen yet).
            min_deliciousness_map[color] = deliciousness

    # After processing all N beans, the min_deliciousness_map contains the minimum
    # deliciousness for each distinct color present in the input. This value represents
    # the minimum possible deliciousness you would get if you chose that specific color.

    # Initialize the variable to store the maximum of these minimum deliciousness values.
    # We want to find the color choice that maximizes this minimum guaranteed deliciousness.
    # Since all deliciousness values A_i are guaranteed to be >= 1, initializing to 0 is safe.
    # If N=0 (though the constraint is N>=1), the map would be empty, and 0 would be printed correctly.
    max_of_mins = 0

    # Check if the map is not empty (i.e., if N > 0 and there were beans processed)
    if min_deliciousness_map:
        # If there are entries in the map, find the maximum value among all the stored minimums.
        # `min_deliciousness_map.values()` gives an iterable of all the minimum deliciousness values
        # (one for each distinct color). `max()` finds the largest among them.
        # This largest value is the answer: the maximum possible value for the minimum deliciousness
        # you can achieve by choosing the optimal color.
        max_of_mins = max(min_deliciousness_map.values())

    # Print the final result (the maximum of the minimums) to standard output.
    print(max_of_mins)

# Execute the solve function to run the main logic of the program when the script is executed.
solve()