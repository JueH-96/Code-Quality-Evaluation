# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
S = [input() for _ in range(N)]

for i in range(N):
    score = 0
    for j in range(M):
        if S[i][j] == 'o':
            score += A[j]
    score += i + 1
    
    ans = 0
    unsolved = []
    for j in range(M):
        if S[i][j] == 'x':
            unsolved.append(A[j])
    unsolved.sort(reverse=True)
    
    others = []
    for k in range(N):
        if k != i:
            other_score = 0
            for j in range(M):
                if S[k][j] == 'o':
                    other_score += A[j]
            other_score += k + 1
            others.append(other_score)
    
    
    cur_score = score
    
    for j in range(len(unsolved)):
        cur_score += unsolved[j]
        exceed = True
        for other_score in others:
            if cur_score <= other_score:
                exceed = False
                break
        if exceed:
            ans = j + 1
            break

    print(ans)