# YOUR CODE HERE
import itertools

# Read input
a, b, c, d, e = map(int, input().split())

# Generate all non-empty subsequences of "ABCDE"
subsequences = []
for r in range(1, 6):
    subsequences.extend(itertools.combinations("ABCDE", r))

# Calculate scores for each subsequence
scores = []
for seq in subsequences:
    score = sum(globals()[char.lower()] for char in seq)
    scores.append((score, ''.join(seq)))

# Sort by score (descending) and lexicographically
scores.sort(key=lambda x: (-x[0], x[1]))

# Print the names of participants in order of their scores
for score, name in scores:
    print(name)