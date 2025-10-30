# YOUR CODE HERE
from collections import Counter

# Read input
cards = list(map(int, input().split()))

# Count the frequency of each card
freq = Counter(cards)

# Determine if adding one card can form a Full House
# There are two possible scenarios:
# 1. Three cards are the same, and the fourth is different. Adding the same as the majority forms a Full House.
# 2. Two pairs of cards. Adding one of the pairs forms a Full House.

# Scenario 1: Three of a kind and one different
if len(freq) == 2:
    counts = list(freq.values())
    if (counts[0] == 3 and counts[1] == 1) or (counts[0] == 1 and counts[1] == 3):
        print("Yes")
    else:
        print("No")
# Scenario 2: Two pairs
elif len(freq) == 2:
    counts = list(freq.values())
    if counts[0] == 2 and counts[1] == 2:
        print("Yes")
    else:
        print("No")
# Scenario 3: All four cards are the same
elif len(freq) == 1:
    print("No")
# Scenario 4: All cards are different
elif len(freq) == 4:
    print("No")
else:
    print("No")