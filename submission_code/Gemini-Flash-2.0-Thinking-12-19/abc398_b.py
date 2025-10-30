# YOUR CODE HERE
import sys
from collections import Counter

# Read the input line from stdin, split by space, and convert to integers
# The input consists of 7 space-separated integers A_1 ... A_7.
line = sys.stdin.readline().split()
cards = list(map(int, line))

# Count the frequency of each card rank using collections.Counter.
# This gives us a dictionary-like object mapping each rank (integer)
# to the number of times it appears in the 7 cards.
counts = Counter(cards)

# A set of five cards forms a full house if there are three cards of one rank
# and two cards of a different rank.
# To be able to choose five cards from the seven input cards to form a full house,
# there must exist two distinct ranks, say rank_x and rank_y, among the input cards,
# such that the input set contains at least 3 cards of rank_x and at least 2 cards of rank_y.

# Flag to track if such a combination of ranks is found.
found_full_house = False

# Iterate through all distinct ranks that are present in the input cards.
# We consider each present rank as a potential candidate for the rank
# that appears three times in the full house (the "three-of-a-kind").
for rank_for_three in counts:
    # Check if the current rank appears at least 3 times in the input set.
    if counts[rank_for_three] >= 3:
        # If we found a rank with count >= 3, we now need to check if there is
        # a *different* rank present in the input set that appears at least 2 times.
        # This second rank would be the candidate for the pair in the full house.
        for rank_for_pair in counts:
            # Check two conditions:
            # 1. The second rank must be different from the first rank (rank_for_three).
            # 2. The second rank must appear at least 2 times in the input set.
            if rank_for_pair != rank_for_three and counts[rank_for_pair] >= 2:
                # If both conditions are met, we have found a combination of two distinct ranks
                # (rank_for_three and rank_for_pair) with sufficient counts (>=3 and >=2 respectively)
                # in the original 7 cards. This means it is possible to choose 3 cards of rank_for_three
                # and 2 cards of rank_for_pair from the original set to form a full house (5 cards total).
                found_full_house = True
                # Since we only need to determine if it is *possible*, finding just one such
                # combination is enough. We can stop searching.
                break # Exit the inner loop
    # If found_full_house is True, it means we found a valid combination in the inner loop,
    # so we can exit the outer loop as well.
    if found_full_house:
        break

# Print the final result to stdout.
# If found_full_house is True, it is possible to form a full house.
# Otherwise, it is not possible.
if found_full_house:
    sys.stdout.write("Yes
")
else:
    sys.stdout.write("No
")