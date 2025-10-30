# YOUR CODE HERE
# Read input
cards = list(map(int, input().split()))

# Count frequencies
freq = {}
for card in cards:
    freq[card] = freq.get(card, 0) + 1

# Check if we can form a full house
for x in freq:
    if freq[x] >= 3:
        for y in freq:
            if x != y and freq[y] >= 2:
                print("Yes")
                exit()

print("No")