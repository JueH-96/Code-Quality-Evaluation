# YOUR CODE HERE
from itertools import combinations

# Read input
scores = list(map(int, input().split()))
a, b, c, d, e = scores

# Map problems to their scores
problem_scores = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}

# Generate all non-empty subsequences of "ABCDE"
problems = "ABCDE"
participants = []

# Generate all possible combinations of length 1 to 5
for length in range(1, 6):
    for combo in combinations(problems, length):
        name = ''.join(combo)
        # Calculate total score for this participant
        total_score = sum(problem_scores[problem] for problem in name)
        participants.append((total_score, name))

# Sort by score (descending) then by name (ascending)
# Using negative score for descending order
participants.sort(key=lambda x: (-x[0], x[1]))

# Output the names
for score, name in participants:
    print(name)