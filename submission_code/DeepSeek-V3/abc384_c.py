# YOUR CODE HERE
a, b, c, d, e = map(int, input().split())
problems = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}

# Generate all non-empty subsequences of 'ABCDE'
from itertools import combinations

subsequences = []
for r in range(1, 6):
    for combo in combinations('ABCDE', r):
        subsequences.append(''.join(combo))

# Calculate scores for each subsequence
scores = []
for sub in subsequences:
    score = sum(problems[char] for char in sub)
    scores.append((score, sub))

# Sort by score descending, then lexicographically
scores.sort(reverse=True, key=lambda x: (x[0], x[1]))

# Print the sorted subsequences
for score, sub in scores:
    print(sub)