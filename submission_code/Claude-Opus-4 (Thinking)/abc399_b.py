# YOUR CODE HERE
n = int(input())
scores = list(map(int, input().split()))

# Initialize ranks (None means undetermined)
ranks = [None] * n
r = 1

# Continue until all ranks are determined
while any(rank is None for rank in ranks):
    # Find the maximum score among undetermined people
    max_score = -1
    for i in range(n):
        if ranks[i] is None and scores[i] > max_score:
            max_score = scores[i]
    
    # Count and assign ranks to all people with this maximum score
    count = 0
    for i in range(n):
        if ranks[i] is None and scores[i] == max_score:
            ranks[i] = r
            count += 1
    
    # Update r
    r += count

# Print the ranks
for rank in ranks:
    print(rank)