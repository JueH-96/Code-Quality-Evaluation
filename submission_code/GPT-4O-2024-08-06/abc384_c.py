# YOUR CODE HERE
from itertools import combinations

def solve():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    # Read the scores for problems A, B, C, D, E
    a, b, c, d, e = map(int, data.split())
    
    # Problem scores dictionary
    scores = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}
    
    # All possible non-empty subsequences of "ABCDE"
    problems = "ABCDE"
    subsequences = []
    
    # Generate all non-empty subsequences
    for length in range(1, len(problems) + 1):
        for combo in combinations(problems, length):
            subsequences.append(''.join(combo))
    
    # Calculate scores for each subsequence
    participant_scores = []
    for subseq in subsequences:
        score = sum(scores[char] for char in subseq)
        participant_scores.append((score, subseq))
    
    # Sort by score descending, then lexicographically
    participant_scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Output the sorted participant names
    for _, name in participant_scores:
        print(name)