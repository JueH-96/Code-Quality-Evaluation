a, b, c, d, e = map(int, input().split())

def generate_subsequences(s):
    subsequences = []
    n = len(s)
    # Generate all non-empty subsets using bit manipulation
    for mask in range(1, 1 << n):
        subseq = ""
        for i in range(n):
            if mask & (1 << i):
                subseq += s[i]
        subsequences.append(subseq)
    return subsequences

def calculate_score(name):
    score = 0
    scores = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}
    for char in name:
        score += scores[char]
    return score

# Generate all participants (subsequences of "ABCDE")
participants = generate_subsequences("ABCDE")

# Calculate scores for each participant
participant_scores = [(name, calculate_score(name)) for name in participants]

# Sort by score (descending) then by name (ascending) for ties
participant_scores.sort(key=lambda x: (-x[1], x[0]))

# Print results
for name, score in participant_scores:
    print(name)