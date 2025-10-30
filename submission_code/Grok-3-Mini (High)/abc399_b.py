import sys

# Read input from stdin
data = sys.stdin.read().split()
N = int(data[0])
P = [int(x) for x in data[1:N+1]]

# Initialize ranks list and set of undetermined indices
ranks = [0] * N
undetermined = set(range(N))
r = 1

# Simulate the ranking process
while undetermined:
    # Find the maximum score among undetermined people
    max_score = max(P[idx] for idx in undetermined)
    
    # Find all indices with this maximum score
    to_assign = [idx for idx in undetermined if P[idx] == max_score]
    k = len(to_assign)
    
    # Assign the current rank to these people
    for idx in to_assign:
        ranks[idx] = r
    
    # Remove these indices from the undetermined set
    undetermined.difference_update(to_assign)
    
    # Increment the rank by k
    r += k

# Output the rank of each person on a separate line
for rank in ranks:
    print(rank)