from collections import Counter

def can_form_full_house(cards):
    # Count the frequency of each card value
    counter = Counter(cards)
    
    # If there are exactly 2 different values across our 4 cards,
    # we can form a Full House by adding the right card
    if len(counter) == 2:
        return "Yes"
    else:
        return "No"

# Read input
cards = list(map(int, input().split()))

# Print the result
print(can_form_full_house(cards))