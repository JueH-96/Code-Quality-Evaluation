# YOUR CODE HERE
from collections import Counter

# Read input
cards = list(map(int, input().split()))

# Count frequency of each card value
freq = Counter(cards)

# Check if we can form a full house
can_form_full_house = False

# We need to find two different values x and y where:
# - x appears at least 3 times
# - y appears at least 2 times
# - x != y

for x in freq:
    if freq[x] >= 3:
        for y in freq:
            if x != y and freq[y] >= 2:
                can_form_full_house = True
                break
    if can_form_full_house:
        break

if can_form_full_house:
    print("Yes")
else:
    print("No")