# YOUR CODE HERE
s_ab, s_ac, s_bc = input().split()

# Initialize scores for each brother
scores = {'A': 0, 'B': 0, 'C': 0}

# Process A vs B comparison
if s_ab == '>':
    scores['A'] += 1
else:  # s_ab == '<'
    scores['B'] += 1

# Process A vs C comparison
if s_ac == '>':
    scores['A'] += 1
else:  # s_ac == '<'
    scores['C'] += 1

# Process B vs C comparison
if s_bc == '>':
    scores['B'] += 1
else:  # s_bc == '<'
    scores['C'] += 1

# Find the brother with score 1 (middle)
for brother, score in scores.items():
    if score == 1:
        print(brother)
        break