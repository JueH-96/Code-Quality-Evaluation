# YOUR CODE HERE
import collections
import sys

# Read the four integer inputs from standard input
# The input line contains four space-separated integers
line = sys.stdin.readline().split()
a, b, c, d = map(int, line)

# Store the four cards in a list
cards = [a, b, c, d]

# Use collections.Counter to count the frequency of each card value
# This gives a dictionary-like object where keys are card values and values are their counts
counts = collections.Counter(cards)

# Get the counts (frequencies) of the distinct card values as a list
count_values = list(counts.values())

# To easily compare the distribution of counts, sort the list of counts.
# This makes the comparison order-independent.
count_values.sort()

# A Full House of 5 cards has counts {3, 2} for two distinct values.
# We start with 4 cards and add one more. The total number of cards will be 5.
# For the final 5 cards to have counts {3, 2}, the original 4 cards must have had counts
# such that adding one card increases the count of one of the existing values to either 3 or 2,
# resulting in the pair {3, 2}.
#
# The possible sorted count distributions for 4 cards that can result in {3, 2} after adding one card are:
# 1. [1, 3]: Adding the card with count 1 makes it {3, 2}.
# 2. [2, 2]: Adding one of the cards makes it {3, 2}.
# Any other initial count distribution for 4 cards ([4], [1, 1, 2], [1, 1, 1, 1]) cannot be turned into {3, 2} by adding just one card.

# Check if the sorted list of counts matches the two possible patterns that allow forming a Full House
if count_values == [1, 3] or count_values == [2, 2]:
    print("Yes")
else:
    print("No")