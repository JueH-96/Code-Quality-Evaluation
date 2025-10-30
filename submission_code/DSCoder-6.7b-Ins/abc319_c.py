from itertools import permutations

# Read inputs
c = [list(map(int, input().split())) for _ in range(3)]

# Initialize the probability
prob = 0

# Iterate over all permutations
for p in permutations(range(1, 10)):
    # Check all rows and columns
    if all(all(c[i][j] != p[3 * i + j] for j in range(3)) for i in range(3)) and \
       all(all(c[i][j] != p[3 * i + j] for i in range(3)) for j in range(3)) and \
       all(p[i * 3 + i] != p[i * 3 + j] for i in range(3)) and \
       all(p[i * 3 + i] != p[i * 3 + j] for i, j in [(0, 2), (1, 1), (2, 0)]):
        prob += 1

# Calculate the probability
prob /= 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2

# Print the probability
print(prob)