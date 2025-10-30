# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
S = []
for _ in range(N):
    S.append(input().strip())

# Calculate current scores
current_scores = []
for i in range(N):
    score = 0
    for j in range(M):
        if S[i][j] == 'o':
            score += A[j]
    score += (i + 1)  # bonus score (player i gets i points, but i is 0-indexed)
    current_scores.append(score)

# For each player
for i in range(N):
    # Find max score among other players
    max_other_score = 0
    for j in range(N):
        if i != j:
            max_other_score = max(max_other_score, current_scores[j])
    
    # If already exceeding all others
    if current_scores[i] > max_other_score:
        print(0)
        continue
    
    # Find unsolved problems and their scores
    unsolved = []
    for j in range(M):
        if S[i][j] == 'x':
            unsolved.append(A[j])
    
    # Sort in descending order (greedy approach - solve highest value problems first)
    unsolved.sort(reverse=True)
    
    # Greedily add problems
    additional_score = 0
    count = 0
    for score in unsolved:
        additional_score += score
        count += 1
        if current_scores[i] + additional_score > max_other_score:
            break
    
    print(count)