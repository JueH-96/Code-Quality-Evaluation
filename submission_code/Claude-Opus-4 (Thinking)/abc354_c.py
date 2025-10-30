# YOUR CODE HERE
n = int(input())
cards = []
for i in range(n):
    a, c = map(int, input().split())
    cards.append((a, c, i+1))  # store strength, cost, and 1-indexed position

# Sort by strength in descending order
cards.sort(reverse=True)

remaining = []
min_cost = float('inf')

for strength, cost, index in cards:
    # Card is kept if no stronger card has lower cost
    if min_cost >= cost:
        remaining.append(index)
    # Update minimum cost seen so far
    min_cost = min(min_cost, cost)

# Sort remaining cards by index and output
remaining.sort()
print(len(remaining))
print(' '.join(map(str, remaining)))