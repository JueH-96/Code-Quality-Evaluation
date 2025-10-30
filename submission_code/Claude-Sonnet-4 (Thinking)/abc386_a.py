from collections import Counter

A, B, C, D = map(int, input().split())
cards = [A, B, C, D]
count = Counter(cards)

# Get the frequency counts and sort them
frequencies = sorted(count.values())

# Check if we can form a Full House by adding one card
# Case 1: [1, 3] - we have 3 of one value and 1 of another
# Case 2: [2, 2] - we have 2 of one value and 2 of another
if frequencies == [1, 3] or frequencies == [2, 2]:
    print("Yes")
else:
    print("No")