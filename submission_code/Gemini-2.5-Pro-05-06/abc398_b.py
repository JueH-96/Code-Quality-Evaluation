import collections

# Read the input cards.
# Example: "1 4 1 4 2 1 3" becomes list of ints [1, 4, 1, 4, 2, 1, 3]
cards = list(map(int, input().split()))

# Count the frequency of each card rank.
# For [1, 4, 1, 4, 2, 1, 3], counts will be Counter({1: 3, 4: 2, 2: 1, 3: 1})
counts = collections.Counter(cards)

found_full_house = False

# Iterate through each unique rank (x_rank) and its count (x_count) from the hand.
# This x_rank is a candidate for the "three of a kind" part of the full house.
for x_rank, x_count in counts.items():
    # We need at least 3 cards of x_rank to form the "three of a kind".
    if x_count >= 3:
        # If we have enough cards for x_rank, look for a y_rank for "two of a kind".
        # Iterate through each unique rank (y_rank) and its count (y_count).
        for y_rank, y_count in counts.items():
            # The ranks x and y must be different for a full house.
            if x_rank == y_rank:
                continue  # Skip if it's the same rank as x_rank.
            
            # We need at least 2 cards of y_rank to form the "two of a kind".
            if y_count >= 2:
                # We found suitable x_rank and y_rank. A full house is possible.
                found_full_house = True
                break  # Exit the inner loop (found a y_rank)
    
    if found_full_house:
        # If the inner loop found a y_rank, a full house is confirmed.
        break  # Exit the outer loop (no need to check other x_ranks)

# Print the result based on whether a full house combination was found.
if found_full_house:
    print("Yes")
else:
    print("No")