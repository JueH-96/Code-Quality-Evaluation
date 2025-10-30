# YOUR CODE HERE
a, b, c, d, e = map(int, input().split())

# Define the problems and their scores
problems = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}

# Generate all non-empty subsequences of 'ABCDE'
from itertools import combinations

subsequences = []
for r in range(1, 6):
    for combo in combinations('ABCDE', r):
        subsequences.append(''.join(combo))

# Calculate the score for each subsequence
scores = []
for sub in subsequences:
    score = 0
    for p in sub:
        score += problems[p]
    scores.append((score, sub))

# Sort first by score in descending order, then lexicographically
scores.sort(reverse=True, key=lambda x: (x[0], x[1]))

# Print the sorted subsequences
for s in scores:
    print(s[1])