# YOUR CODE HERE
from collections import Counter

# Read input
cards = list(map(int, input().split()))

# Count occurrences of each card value
count = Counter(cards)

# Get the frequency distribution
frequencies = sorted(count.values(), reverse=True)

# Check if we can form a Full House by adding one card
can_form = False

# Case 1: We have 3 of one kind and 1 of another
if len(frequencies) == 2 and frequencies == [3, 1]:
    can_form = True

# Case 2: We have 2 of one kind and 2 of another
elif len(frequencies) == 2 and frequencies == [2, 2]:
    can_form = True

# Case 3: We have 2 of one kind, 1 of another, and 1 of yet another
elif len(frequencies) == 3 and frequencies == [2, 1, 1]:
    can_form = True

print("Yes" if can_form else "No")