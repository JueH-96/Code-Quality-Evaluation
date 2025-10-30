# YOUR CODE HERE
from collections import Counter

# Read input
cards = list(map(int, input().split()))

# Count occurrences of each card
card_counts = Counter(cards)

# Check if Full House can be formed by adding one card
if len(card_counts) == 1:
    # All cards are the same, can't form a Full House
    print("No")
elif len(card_counts) == 2:
    # Two types of cards
    if 2 in card_counts.values():
        # Already have a pair, need one more of the other card
        print("Yes")
    else:
        # Have three of a kind, need one more of the other card
        print("Yes")
elif len(card_counts) == 3:
    # Three types of cards
    if 2 in card_counts.values():
        # Have a pair, need one more of that card
        print("Yes")
    else:
        # Have three singles and one pair, can't form a Full House
        print("No")
elif len(card_counts) == 4:
    # All cards are different, can't form a Full House
    print("No")