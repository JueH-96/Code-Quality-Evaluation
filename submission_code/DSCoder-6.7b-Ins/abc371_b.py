import sys

# Read the inputs
N, M = map(int, sys.stdin.readline().split())

# Initialize a list to store the earliest son in each family
earliest_son = [None] * N

# Initialize a list to store the name given to each baby
name_given = []

# Process the babies in chronological order
for i in range(M):
    A_i, B_i = sys.stdin.readline().split()
    A_i = int(A_i) - 1  # Convert to 0-based index

    # If this is the first baby in the family, it is the earliest son
    if earliest_son[A_i] is None:
        earliest_son[A_i] = True
        name_given.append("Yes")
    # If this is not the first baby in the family, it is not the earliest son
    else:
        name_given.append("No")

# Print the name given to each baby
for name in name_given:
    print(name)