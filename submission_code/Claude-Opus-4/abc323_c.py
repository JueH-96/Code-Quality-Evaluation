# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
S = []
for _ in range(N):
    S.append(input().strip())

# Calculate current scores for each player
current_scores = []
for i in range(N):
    score = 0
    for j in range(M):
        if S[i][j] == 'o':
            score += A[j]
    score += (i + 1)  # bonus points (1-indexed)
    current_scores.append(score)

# For each player, find minimum problems needed to exceed all others
for i in range(N):
    # Find the maximum score among other players
    max_other_score = 0
    for j in range(N):
        if i != j:
            max_other_score = max(max_other_score, current_scores[j])
    
    # Find unsolved problems for player i
    unsolved = []
    for j in range(M):
        if S[i][j] == 'x':
            unsolved.append(A[j])
    
    # Sort unsolved problems in descending order (solve highest scoring first)
    unsolved.sort(reverse=True)
    
    # Calculate how many problems needed
    current = current_scores[i]
    problems_needed = 0
    
    for score in unsolved:
        if current > max_other_score:
            break
        current += score
        problems_needed += 1
    
    print(problems_needed)