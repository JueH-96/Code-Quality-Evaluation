# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

scores = []
unsolved = []

for i in range(N):
    S = input()
    score = sum(A[j] for j in range(M) if S[j] == 'o') + i + 1
    scores.append(score)
    unsolved.append([A[j] for j in range(M) if S[j] == 'x'])

for i in range(N):
    current_score = scores[i]
    target_score = max(scores)
    if current_score > target_score:
        print(0)
        continue
    
    unsolved_sorted = sorted(unsolved[i], reverse=True)
    problems_to_solve = 0
    for score in unsolved_sorted:
        current_score += score
        problems_to_solve += 1
        if current_score > target_score:
            break
    
    print(problems_to_solve)