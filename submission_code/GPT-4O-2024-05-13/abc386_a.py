# YOUR CODE HERE
from collections import Counter

def can_form_full_house(cards):
    card_counts = Counter(cards)
    if len(card_counts) == 1:
        # All four cards are the same, adding any card won't form a full house
        return "No"
    elif len(card_counts) == 2:
        # Two different cards, check their counts
        counts = list(card_counts.values())
        if 3 in counts or 2 in counts:
            return "Yes"
        else:
            return "No"
    elif len(card_counts) == 3:
        # Three different cards, check their counts
        counts = list(card_counts.values())
        if 3 in counts or 2 in counts:
            return "Yes"
        else:
            return "No"
    else:
        # Four different cards, adding any card won't form a full house
        return "No"

# Read input
import sys
input = sys.stdin.read().strip()
A, B, C, D = map(int, input.split())

# Determine if a full house can be formed
result = can_form_full_house([A, B, C, D])
print(result)