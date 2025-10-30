import collections

# Read the four card values from standard input
A, B, C, D = map(int, input().split())

# Put the card values into a list
cards = [A, B, C, D]

# Count the frequency of each card value
# e.g., if cards = [7, 7, 7, 1], counts will be Counter({7: 3, 1: 1})
# e.g., if cards = [3, 3, 5, 5], counts will be Counter({3: 2, 5: 2})
counts = collections.Counter(cards)

# Get the list of frequencies (the counts of each distinct card value)
# For Counter({7: 3, 1: 1}), counts.values() is dict_values([3, 1])
# For Counter({3: 2, 5: 2}), counts.values() is dict_values([2, 2])
# Convert to a list and sort it to get a canonical representation.
# For [3, 1], sorted list is [1, 3].
# For [2, 2], sorted list is [2, 2].
freq_values = sorted(list(counts.values()))

# Check if the pattern of frequencies matches one of the conditions for forming a Full House
# Condition 1: Three cards of one kind, one card of another kind (e.g., 7 7 7 1). Frequencies are [1, 3].
#             Adding one more card of the kind that appears once forms a Full House (e.g., add 1 -> 7 7 7 1 1).
# Condition 2: Two cards of one kind, two cards of another kind (e.g., 3 3 5 5). Frequencies are [2, 2].
#             Adding one more card of either kind forms a Full House (e.g., add 3 -> 3 3 3 5 5).
if freq_values == [1, 3] or freq_values == [2, 2]:
    print("Yes")
else:
    # For any other frequency pattern (e.g., [4] for 8 8 8 8; or [1,1,2] for 1 1 3 4),
    # a Full House cannot be formed by adding one card.
    print("No")