# Read input
A, B, C, D = map(int, input().split())

# Count occurrences
from collections import Counter
cards = [A, B, C, D]
count = Counter(cards)

# Try adding each possible card
can_form_full_house = False
for card_to_add in range(1, 14):
    # Create new count
    new_count = count.copy()
    new_count[card_to_add] += 1
    
    # Check if it's a Full House
    # A Full House has exactly 2 distinct values with counts 3 and 2
    if len(new_count) == 2:
        counts = sorted(new_count.values())
        if counts == [2, 3]:
            can_form_full_house = True
            break

# Output result
if can_form_full_house:
    print("Yes")
else:
    print("No")