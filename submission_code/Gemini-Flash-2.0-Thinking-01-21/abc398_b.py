# YOUR CODE HERE
import sys
from collections import Counter

# Read the space-separated integers from standard input
line = sys.stdin.readline().split()
# Convert the list of string numbers to a list of integers
cards = list(map(int, line))

# Use Counter to get the frequency of each card rank
# counts is a dictionary mapping rank to its frequency (number of times it appears)
counts = Counter(cards)

# To form a full house (3 of one rank, 2 of another), from the 7 cards,
# we need to be able to choose 3 cards of some rank 'x' and 2 cards of some distinct rank 'y'.
# This is possible if and only if among the original 7 cards:
# 1. There is at least one rank 'x' with a count of 3 or more.
# 2. There is at least one rank 'y' with a count of 2 or more, where y is different from x.

# Let's find the ranks that appear 3 or more times and those that appear 2 or more times.
ranks_ge_3 = set() # Set of distinct ranks with count >= 3
ranks_ge_2 = set() # Set of distinct ranks with count >= 2

# Iterate through the frequencies of each rank found in the 7 cards
for rank, count in counts.items():
    if count >= 3:
        ranks_ge_3.add(rank)
    if count >= 2:
        ranks_ge_2.add(rank)

# Now, check if we can find two distinct ranks, one for the triple (count >= 3)
# and one for the pair (count >= 2).

# A full house can be formed if the following condition is true:
# (There are at least two distinct ranks with count >= 3)
# OR
# (There is exactly one rank with count >= 3 AND there is at least one other distinct rank with count >= 2)

can_form_full_house = False

# Case 1: There are two or more distinct ranks that appear 3 or more times.
# If len(ranks_ge_3) >= 2, pick any two distinct ranks r1, r2 from ranks_ge_3.
# count(r1) >= 3 and count(r2) >= 3. We can use r1 for the triple and r2 for the pair
# (since count(r2) >= 3 implies count(r2) >= 2). r1 != r2.
if len(ranks_ge_3) >= 2:
    can_form_full_house = True
# Case 2: There is exactly one rank that appears 3 or more times.
# Let this rank be 'triple_rank'. We must use 'triple_rank' for the triple.
# We need a *distinct* rank 'pair_rank' that appears 2 or more times.
# So, we need 'pair_rank' to be in ranks_ge_2 AND 'pair_rank' != 'triple_rank'.
# Since 'triple_rank' is the only element in ranks_ge_3, and ranks_ge_3 is a subset of ranks_ge_2,
# 'triple_rank' is also in ranks_ge_2.
# For ranks_ge_2 to contain an element *other than* 'triple_rank', ranks_ge_2 must have a size of at least 2.
elif len(ranks_ge_3) == 1 and len(ranks_ge_2) >= 2:
    can_form_full_house = True
# If neither of the above cases is met, it's not possible to form a full house.
else:
    can_form_full_house = False

# Print the final result
if can_form_full_house:
    print("Yes")
else:
    print("No")

# END OF YOUR CODE