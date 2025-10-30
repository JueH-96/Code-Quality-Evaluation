# YOUR CODE HERE
from collections import Counter

# Read input
A, B, C, D = map(int, input().split())

# Count the occurrences of each number
count = Counter([A, B, C, D])

# Check if we can form a Full House by adding one card
for num in count:
    if count[num] == 2:
        # If there are two cards with the same number, we need three more cards with the same number
        for other_num in count:
            if other_num != num and count[other_num] >= 2:
                print("Yes")
                exit()
    elif count[num] == 3:
        # If there are three cards with the same number, we need two more cards with the same number
        for other_num in count:
            if other_num != num and count[other_num] >= 1:
                print("Yes")
                exit()

print("No")