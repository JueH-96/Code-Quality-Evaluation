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
    score += (i + 1)  # bonus score
    current_scores.append(score)

# For each player, find minimum problems to solve
for i in range(N):
    # Find max score among other players
    max_other_score = max(current_scores[j] for j in range(N) if j != i)
    
    # If already ahead, answer is 0
    if current_scores[i] > max_other_score:
        print(0)
        continue
    
    # Find unsolved problems for player i
    unsolved = []
    for j in range(M):
        if S[i][j] == 'x':
            unsolved.append(A[j])
    
    # Sort unsolved problems by score in descending order
    unsolved.sort(reverse=True)
    
    # Find minimum number of problems to solve
    current_score = current_scores[i]
    count = 0
    for score in unsolved:
        current_score += score
        count += 1
        if current_score > max_other_score:
            break
    
    print(count)