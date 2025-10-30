def can_form_full_house(cards):
    from collections import Counter
    
    # Count the occurrences of each card
    count = Counter(cards)
    
    # We need at least one value with 3 occurrences and one with 2 occurrences
    three_of_a_kind = 0
    pair = 0
    
    for value in count.values():
        if value >= 3:
            three_of_a_kind += 1
        if value >= 2:
            pair += 1
    
    # Check if we can form a full house
    if three_of_a_kind >= 1 and pair >= 2:
        return "Yes"
    return "No"

# Read input
import sys
input_data = sys.stdin.read().strip()
cards = list(map(int, input_data.split()))

# Output the result
print(can_form_full_house(cards))