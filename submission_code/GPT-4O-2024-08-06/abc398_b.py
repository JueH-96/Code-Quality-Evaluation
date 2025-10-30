# YOUR CODE HERE
def can_form_full_house(cards):
    from collections import Counter
    
    # Count the occurrences of each card number
    card_count = Counter(cards)
    
    # We need exactly one number with count 3 and another with count 2
    has_three = False
    has_two = False
    
    for count in card_count.values():
        if count == 3:
            has_three = True
        elif count == 2:
            has_two = True
    
    # If both conditions are satisfied, we can form a full house
    if has_three and has_two:
        return "Yes"
    else:
        return "No"

# Read input
import sys
input = sys.stdin.read
cards = list(map(int, input().strip().split()))

# Output the result
print(can_form_full_house(cards))