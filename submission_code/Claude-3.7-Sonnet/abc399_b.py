# Read input
N = int(input())
scores = list(map(int, input().split()))

# Initialize ranks array (None means undetermined)
ranks = [None] * N

# Initialize the rank variable
r = 1

# Continue until all ranks are determined
while None in ranks:
    # Find maximum score among people with undetermined ranks
    max_score = max(scores[i] for i in range(N) if ranks[i] is None)
    
    # Count people with this max score and assign rank
    k = 0
    for i in range(N):
        if ranks[i] is None and scores[i] == max_score:
            ranks[i] = r
            k += 1
    
    # Update r for next iteration
    r += k

# Print ranks
for rank in ranks:
    print(rank)