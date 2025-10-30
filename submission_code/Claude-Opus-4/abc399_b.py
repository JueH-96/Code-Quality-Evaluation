# YOUR CODE HERE
n = int(input())
scores = list(map(int, input().split()))

# Create a list to store ranks (0-indexed, but we'll use 1-indexed logic)
ranks = [0] * n

# Track which people have been ranked
ranked = [False] * n

r = 1  # Current rank to assign

while not all(ranked):
    # Find maximum score among unranked people
    max_score = -1
    for i in range(n):
        if not ranked[i] and scores[i] > max_score:
            max_score = scores[i]
    
    # Count how many people have this max score
    k = 0
    for i in range(n):
        if not ranked[i] and scores[i] == max_score:
            k += 1
    
    # Assign rank r to all people with max_score
    for i in range(n):
        if not ranked[i] and scores[i] == max_score:
            ranks[i] = r
            ranked[i] = True
    
    # Update r
    r += k

# Print the ranks
for rank in ranks:
    print(rank)