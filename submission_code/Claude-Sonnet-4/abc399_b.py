# YOUR CODE HERE
n = int(input())
scores = list(map(int, input().split()))

# Create list of (score, index) pairs
people = [(scores[i], i) for i in range(n)]

# Initialize ranks array
ranks = [0] * n

# Initialize rank counter
r = 1

# Keep track of which people have been assigned ranks
assigned = [False] * n

while not all(assigned):
    # Find maximum score among unassigned people
    max_score = -1
    for i in range(n):
        if not assigned[i] and scores[i] > max_score:
            max_score = scores[i]
    
    # Count how many people have this max score
    count = 0
    for i in range(n):
        if not assigned[i] and scores[i] == max_score:
            count += 1
    
    # Assign rank r to all people with max_score
    for i in range(n):
        if not assigned[i] and scores[i] == max_score:
            ranks[i] = r
            assigned[i] = True
    
    # Update rank counter
    r += count

# Print results
for rank in ranks:
    print(rank)