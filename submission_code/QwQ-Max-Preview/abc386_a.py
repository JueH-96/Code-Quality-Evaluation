import sys
from collections import Counter

# Read the input values
a, b, c, d = map(int, sys.stdin.readline().split())
original = Counter([a, b, c, d])

# Check each possible card to add (1 to 13)
for x in range(1, 14):
    new_counts = original.copy()
    new_counts[x] += 1  # Add the new card
    # Get the sorted list of counts in descending order
    counts = sorted(new_counts.values(), reverse=True)
    if counts == [3, 2]:
        print("Yes")
        sys.exit()

# If no valid card found
print("No")