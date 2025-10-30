# YOUR CODE HERE
from collections import Counter

# Read input
cards = list(map(int, input().split()))

# Count the frequency of each card
freq = Counter(cards)

# Determine if adding one card can form a Full House
# Full House requires either:
# 1. Three of one kind and two of another
# 2. Two of one kind and three of another

# Check if the current counts can form a Full House with one additional card
# We need to have either:
# - Three of one kind and one of another, and the additional card is the same as the one with one count
# - Two of one kind and two of another, and the additional card is the same as one of the two kinds

# Get the counts of each card
counts = list(freq.values())

# Sort the counts to make it easier to check
counts.sort()

# Check the possible scenarios
if counts == [1, 3]:
    # We have one card with count 1 and three with count 3
    # Adding one more of the card with count 1 will make it two
    # So, we will have three of one kind and two of another
    print("Yes")
elif counts == [2, 2]:
    # We have two cards with count 2 each
    # Adding one more of either will make it three of one kind and two of another
    print("Yes")
elif counts == [1, 1, 2]:
    # We have two cards with count 1 and one with count 2
    # Adding one more of the card with count 2 will make it three of one kind and two of another
    print("Yes")
elif counts == [1, 1, 1, 1]:
    # We have four distinct cards
    # Adding one more of any will not form a Full House
    print("No")
elif counts == [4]:
    # We have four of the same card
    # Adding one more will make five of the same, which is not a Full House
    print("No")
else:
    # Other cases are not possible
    print("No")