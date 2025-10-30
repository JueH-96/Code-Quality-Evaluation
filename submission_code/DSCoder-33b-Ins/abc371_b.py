# YOUR CODE HERE

# Read the inputs
N, M = map(int, input().split())
babies = []
for _ in range(M):
    A, B = input().split()
    babies.append((int(A), B))

# Initialize a dictionary to keep track of the earliest born boy in each family
earliest_boys = {i: None for i in range(1, N+1)}

# Iterate over the babies
for A, B in babies:
    # If the baby is male
    if B == 'M':
        # If this is the earliest born boy in this family
        if earliest_boys[A] is None:
            earliest_boys[A] = B

# Iterate over the babies again
for A, B in babies:
    # If the baby is male and is the earliest born boy in its family
    if B == 'M' and earliest_boys[A] == B:
        print('Yes')
        # Update the earliest born boy in this family
        earliest_boys[A] = None
    else:
        print('No')