# YOUR CODE HERE
import sys
from collections import Counter

# Read the line from standard input
line = sys.stdin.readline().strip()

# Split the line into numbers and convert them to integers
cards = list(map(int, line.split()))

# Count the frequency of each card value
# collections.Counter is perfect for this
counts = Counter(cards)

# To form a full house (3 of one rank, 2 of another),
# the original 7 cards must contain enough cards to form this combination.
# Specifically, there must exist two different card values, say x and y (x != y),
# such that the count of x in the seven cards is at least 3,
# and the count of y in the seven cards is at least 2.

possible = False

# Iterate through each unique card value and its count.
# We check if this value could be the one forming the triplet (>= 3 cards).
for value_x, count_x in counts.items():
    # If the count of value_x is 3 or more, it's a potential triplet value.
    if count_x >= 3:
        # Now that we have a potential triplet value (value_x),
        # we need to check if there's a DIFFERENT value (value_y)
        # that has a count of 2 or more, which could form the pair.
        for value_y, count_y in counts.items():
            # We must ensure value_y is different from value_x (x != y)
            # And check if the count of value_y is 2 or more.
            if value_y != value_x and count_y >= 2:
                # We found a value_x with count >= 3 and a different value_y
                # with count >= 2. This means we can pick 3 cards of value_x
                # and 2 cards of value_y from the original 7 cards to form a full house.
                possible = True
                # Since we only need to find one such combination of (x, y),
                # we can stop searching once we find the first one.
                break # Exit the inner loop (searching for a pair value_y)
    # If we found a full house possibility in the inner loop (possible is True),
    # we can also stop searching through the potential triplet values (value_x).
    if possible:
        break # Exit the outer loop (searching for a triplet value_x)

# Print the final result based on whether a full house was found to be possible.
if possible:
    print("Yes")
else:
    print("No")