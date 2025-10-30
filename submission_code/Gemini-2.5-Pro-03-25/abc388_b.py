# YOUR CODE HERE
import sys

# Read N (number of snakes) and D (max length increase) from the first line of input
# N represents the number of snakes, D represents the maximum integer increase in length.
n, d = map(int, sys.stdin.readline().split())

# Read the details for each of the N snakes.
# Each snake has an initial thickness T_i and initial length L_i.
# Store these details in a list of lists (or list of tuples).
snakes_data = []
for _ in range(n):
    # Read the thickness and length for the current snake
    t_i, l_i = map(int, sys.stdin.readline().split())
    # Append the [thickness, length] pair to our list
    snakes_data.append([t_i, l_i])

# Iterate through each possible integer length increase k, from 1 up to D (inclusive).
for k in range(1, d + 1):
    # For each value of k, we need to find the maximum weight among all snakes.
    # Initialize the maximum weight found so far for this specific k to 0.
    # Since T_i, L_i, and k are all positive, the weight will be positive,
    # so 0 is a safe initial value for the maximum.
    current_max_weight = 0

    # Iterate through each snake to calculate its weight when its length is increased by k.
    for i in range(n):
        # Get the thickness (T_i) and initial length (L_i) for the i-th snake.
        t_i = snakes_data[i][0]
        l_i = snakes_data[i][1]

        # Calculate the new length of the snake after increasing by k.
        new_length = l_i + k

        # Calculate the weight of the snake, which is thickness * new_length.
        weight = t_i * new_length

        # Update the maximum weight found for this k if the current snake's weight is greater.
        # The max() function conveniently returns the larger of its two arguments.
        current_max_weight = max(current_max_weight, weight)

    # After checking all snakes for the current value of k,
    # print the maximum weight found.
    # The output requires printing one line per value of k.
    print(current_max_weight)