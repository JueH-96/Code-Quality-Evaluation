import sys

# Read input from stdin
data = sys.stdin.read().split()
a, b, c, d, e = map(int, data)

# Define letters and scores
letters = ['A', 'B', 'C', 'D', 'E']
scores_list = [a, b, c, d, e]

# List to hold participants as tuples of (score, name)
participants = []

# Generate all 31 participants using bit masks from 1 to 31
for mask in range(1, 32):
    name = ''
    score_sum = 0
    for i in range(5):
        if mask & (1 << i):
            name += letters[i]
            score_sum += scores_list[i]
    participants.append((score_sum, name))

# Sort participants: descending order of score, ascending lexicographical order of name
sorted_participants = sorted(participants, key=lambda x: (-x[0], x[1]))

# Output the names of the participants in the sorted order
for score, name in sorted_participants:
    print(name)