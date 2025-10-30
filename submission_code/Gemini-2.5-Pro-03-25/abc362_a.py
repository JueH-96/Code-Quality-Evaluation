# YOUR CODE HERE
import sys

# Read R, G, B costs from the first line of standard input
# R: cost of red pen
# G: cost of green pen
# B: cost of blue pen
r, g, b = map(int, sys.stdin.readline().split())

# Read the disliked color C from the second line of standard input
c = sys.stdin.readline().strip() # Use strip() to remove potential trailing newline

# Determine the minimum cost based on the disliked color C
# Initialize min_cost variable
min_cost = 0

if c == 'Red':
    # If Red is disliked, Takahashi can only buy Green or Blue pens.
    # The minimum cost is the lesser of the Green and Blue pen costs.
    min_cost = min(g, b)
elif c == 'Green':
    # If Green is disliked, Takahashi can only buy Red or Blue pens.
    # The minimum cost is the lesser of the Red and Blue pen costs.
    min_cost = min(r, b)
elif c == 'Blue':
    # If Blue is disliked, Takahashi can only buy Red or Green pens.
    # The minimum cost is the lesser of the Red and Green pen costs.
    min_cost = min(r, g)
# The problem statement guarantees C will be one of "Red", "Green", or "Blue".

# Print the calculated minimum cost to standard output
print(min_cost)

# YOUR CODE HERE