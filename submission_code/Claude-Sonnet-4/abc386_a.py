# YOUR CODE HERE
from collections import Counter

# Read input
A, B, C, D = map(int, input().split())

# Count frequency of each card value
cards = [A, B, C, D]
count = Counter(cards)

# Get the frequency values
frequencies = list(count.values())
frequencies.sort()

# Check if we can form a Full House by adding one card
# Case 1: We have 3 of one kind and 1 of another (3,1)
# Case 2: We have 2 of one kind and 2 of another (2,2)

if frequencies == [1, 3] or frequencies == [2, 2]:
    print("Yes")
else:
    print("No")