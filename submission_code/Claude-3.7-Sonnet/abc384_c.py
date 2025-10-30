def generate_subsequences(s):
    """Generate all non-empty subsequences of string s."""
    subsequences = []
    n = len(s)
    # Generate all 2^n - 1 non-empty subsequences
    for i in range(1, 2**n):
        subsequence = ""
        for j in range(n):
            if (i & (1 << j)) > 0:
                subsequence += s[j]
        subsequences.append(subsequence)
    return subsequences

def calculate_score(participant, scores):
    """Calculate the score for a participant."""
    total_score = 0
    problem_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    for problem in participant:
        total_score += scores[problem_to_index[problem]]
    return total_score

# Read input
a, b, c, d, e = map(int, input().split())
scores = [a, b, c, d, e]

# Generate all participants (non-empty subsequences of "ABCDE")
participants = generate_subsequences("ABCDE")

# Calculate scores for each participant
participant_scores = [(participant, calculate_score(participant, scores)) for participant in participants]

# Sort by score (decreasing) and then lexicographically
participant_scores.sort(key=lambda x: (-x[1], x[0]))

# Output
for participant, _ in participant_scores:
    print(participant)