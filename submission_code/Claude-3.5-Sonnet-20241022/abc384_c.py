# Read input scores
a, b, c, d, e = map(int, input().split())

# Generate all possible non-empty subsequences of "ABCDE"
def get_all_subsequences():
    base = "ABCDE"
    subsequences = []
    for i in range(1, 1 << len(base)):
        seq = ""
        for j in range(len(base)):
            if i & (1 << j):
                seq += base[j]
        subsequences.append(seq)
    return subsequences

# Calculate score for a participant based on their name
def calculate_score(name):
    score = 0
    if 'A' in name: score += a
    if 'B' in name: score += b
    if 'C' in name: score += c
    if 'D' in name: score += d
    if 'E' in name: score += e
    return score

# Get all participants and their scores
participants = get_all_subsequences()
participant_scores = [(name, calculate_score(name)) for name in participants]

# Sort participants by score (descending) and name (lexicographically)
participant_scores.sort(key=lambda x: (-x[1], x[0]))

# Print results
for participant, _ in participant_scores:
    print(participant)