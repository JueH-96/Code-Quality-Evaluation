# YOUR CODE HERE
from collections import Counter

# Read input
cards = list(map(int, input().split()))

# Count frequency of each card value
count = Counter(cards)

# Get the frequencies (how many times each value appears)
frequencies = list(count.values())
frequencies.sort(reverse=True)

# Check if we can form a full house
# We need at least one value with 3+ occurrences and another with 2+ occurrences
can_form_full_house = False

# Check all possible combinations
for i in range(len(frequencies)):
    if frequencies[i] >= 3:  # This value can provide the "three of a kind"
        for j in range(len(frequencies)):
            if i != j and frequencies[j] >= 2:  # Different value can provide the "pair"
                can_form_full_house = True
                break
        if can_form_full_house:
            break

if can_form_full_house:
    print("Yes")
else:
    print("No")