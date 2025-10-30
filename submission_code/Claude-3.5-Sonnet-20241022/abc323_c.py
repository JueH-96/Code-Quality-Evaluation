N, M = map(int, input().split())
A = list(map(int, input().split()))
S = [input() for _ in range(N)]

for i in range(N):
    # Calculate current score for player i
    curr_score = i + 1  # bonus score
    unsolved = []
    for j in range(M):
        if S[i][j] == 'o':
            curr_score += A[j]
        else:
            unsolved.append(A[j])
    
    # Calculate max scores of other players
    max_other_score = 0
    for j in range(N):
        if j == i:
            continue
        score = j + 1  # bonus score
        for k in range(M):
            if S[j][k] == 'o':
                score += A[k]
        max_other_score = max(max_other_score, score)
    
    # Sort unsolved problems in descending order
    unsolved.sort(reverse=True)
    
    # Find minimum problems needed
    problems_needed = 0
    temp_score = curr_score
    
    if temp_score > max_other_score:
        print(0)
        continue
        
    for score in unsolved:
        problems_needed += 1
        temp_score += score
        if temp_score > max_other_score:
            break
            
    print(problems_needed)