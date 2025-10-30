from collections import Counter

# Read input
cards = list(map(int, input().split()))

# Count frequencies
freq = Counter(cards)

# Check if we can form a full house
can_form = False

for val1, count1 in freq.items():
    if count1 >= 3:
        for val2, count2 in freq.items():
            if val1 != val2 and count2 >= 2:
                can_form = True
                break
        if can_form:
            break

print("Yes" if can_form else "No")