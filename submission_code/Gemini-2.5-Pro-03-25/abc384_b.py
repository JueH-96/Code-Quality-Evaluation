# YOUR CODE HERE
import sys

# Read N and initial rating R from the first line of input
n, r = map(int, sys.stdin.readline().split())

# Initialize the current rating with the initial rating R
current_rating = r

# Process each of the N contests
for _ in range(n):
    # Read the contest division D and performance A for the current contest
    # D indicates the division (1 or 2)
    # A indicates the performance change if applicable
    d, a = map(int, sys.stdin.readline().split())

    # Flag to determine if a rating update should occur for this contest
    update = False

    # Check the conditions for rating update based on the contest division D
    # and the participant's current rating `current_rating` at the start of the contest.
    if d == 1:
        # For ARC Div. 1, rating update applies if the rating is between 1600 and 2799, inclusive.
        if 1600 <= current_rating <= 2799:
            update = True
    elif d == 2:
        # For ARC Div. 2, rating update applies if the rating is between 1200 and 2399, inclusive.
        # The problem constraints guarantee D is either 1 or 2.
        if 1200 <= current_rating <= 2399:
            update = True

    # If the participant is subject to a rating update in this contest,
    # modify the current rating by adding the performance value A.
    if update:
        current_rating += a

# After processing all N contests, print the final rating.
print(current_rating)