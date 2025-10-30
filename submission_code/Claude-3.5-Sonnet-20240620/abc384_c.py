# YOUR CODE HERE
from itertools import combinations

def generate_participants(problems):
    return [''.join(combo) for i in range(1, len(problems) + 1) for combo in combinations(problems, i)]

def calculate_score(participant, scores):
    return sum(scores[problems.index(problem)] for problem in participant)

def main():
    # Read input
    scores = list(map(int, input().split()))
    
    # Define problems and generate participants
    problems = "ABCDE"
    participants = generate_participants(problems)
    
    # Calculate scores for each participant
    participant_scores = [(p, calculate_score(p, scores)) for p in participants]
    
    # Sort participants by score (descending) and then by name (lexicographically)
    sorted_participants = sorted(participant_scores, key=lambda x: (-x[1], x[0]))
    
    # Print results
    for participant, _ in sorted_participants:
        print(participant)

if __name__ == "__main__":
    main()