from itertools import combinations

# Read the number of people N and the number of photos M
N, M = map(int, input().split())

# Initialize a set to keep track of pairs that have stood next to each other
adjacent_pairs = set()

# Read each photo and update the set of adjacent pairs
for _ in range(M):
    photo = list(map(int, input().split()))
    for i in range(N - 1):
        adjacent_pairs.add((photo[i], photo[i + 1]))
        adjacent_pairs.add((photo[i + 1], photo[i]))

# Calculate the total number of possible pairs
total_pairs = set(combinations(range(1, N + 1), 2))

# Calculate the number of pairs that have never stood next to each other
non_adjacent_pairs = total_pairs - adjacent_pairs

# Print the number of such pairs
print(len(non_adjacent_pairs))