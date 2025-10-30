# Read input
cards = list(map(int, input().split()))

# Count frequency of each number
freq = {}
for card in cards:
    freq[card] = freq.get(card, 0) + 1

# Check if full house is possible
def can_form_full_house():
    # We need at least two different numbers
    if len(freq) < 2:
        return False
    
    # Check all possible combinations of numbers
    # One number should appear at least 3 times and another at least 2 times
    for x in freq:
        for y in freq:
            if x != y:  # Numbers must be different
                if freq[x] >= 3 and freq[y] >= 2:
                    return True
    return False

# Print result
print("Yes" if can_form_full_house() else "No")