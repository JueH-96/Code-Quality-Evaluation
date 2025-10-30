import itertools
import sys

input = sys.stdin.read
data = input().split()

# Scores for each problem
a, b, c, d, e = map(int, data)
scores = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}

# Generate all non-empty subsequences of "ABCDE"
problems = "ABCDE"
participants = []
for i in range(1, len(problems) + 1):
    for subset in itertools.combinations(problems, i):
        participants.append(''.join(subset))

# Calculate the score for each participant
participant_scores = []
for participant in participants:
    score = sum(scores[problem] for problem in participant)
    participant_scores.append((participant, score))

# Sort participants by score (descending) and lexicographically (ascending)
participant_scores.sort(key=lambda x: (-x[1], x[0]))

# Output the sorted participants
for participant, score in participant_scores:
    print(participant)