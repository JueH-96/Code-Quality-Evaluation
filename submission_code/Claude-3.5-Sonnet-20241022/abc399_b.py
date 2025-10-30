# Read input
N = int(input())
P = list(map(int, input().split()))

# Create list of indices and scores
scores = [(score, i) for i, score in enumerate(P)]
ranks = [0] * N  # Store ranks for each person

# Sort scores in descending order
scores.sort(reverse=True)

current_rank = 1
i = 0
while i < N:
    # Find all people with the same score
    current_score = scores[i][0]
    same_score_count = 0
    
    # Count how many people have the same score
    while i + same_score_count < N and scores[i + same_score_count][0] == current_score:
        # Assign current rank to all people with this score
        person_index = scores[i + same_score_count][1]
        ranks[person_index] = current_rank
        same_score_count += 1
    
    # Move to next group of scores
    i += same_score_count
    # Update rank for next iteration
    current_rank += same_score_count

# Print ranks
for rank in ranks:
    print(rank)