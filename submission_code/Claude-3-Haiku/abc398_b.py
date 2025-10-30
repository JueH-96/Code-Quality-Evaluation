from collections import Counter

# Read input from stdin
input_str = input()
card_values = [int(x) for x in input_str.split()]

# Count the frequency of each card value
card_counts = Counter(card_values)

# Check if a full house can be formed
for x, count_x in card_counts.items():
    for y, count_y in card_counts.items():
        if x != y and count_x == 3 and count_y == 2:
            print("Yes")
            return

print("No")