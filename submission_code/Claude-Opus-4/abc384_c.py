# YOUR CODE HERE
import itertools

# Read the scores
scores = list(map(int, input().split()))
a, b, c, d, e = scores

# Map problems to their scores
problem_scores = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}

# Generate all non-empty subsequences of ABCDE
participants = []
problems = "ABCDE"

# Generate all possible combinations
for r in range(1, 6):  # 1 to 5 problems
    for combo in itertools.combinations(problems, r):
        participant_name = ''.join(combo)
        participant_score = sum(problem_scores[p] for p in participant_name)
        participants.append((participant_name, participant_score))

# Sort by score (descending) and then by name (lexicographically)
participants.sort(key=lambda x: (-x[1], x[0]))

# Print the names
for name, score in participants:
    print(name)