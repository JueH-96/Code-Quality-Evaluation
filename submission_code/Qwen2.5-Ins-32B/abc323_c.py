import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:M+2]))
    S = data[M+2:]
    
    scores = []
    for i in range(N):
        score = sum(A[j] for j in range(M) if S[i][j] == 'o') + (i + 1)
        scores.append(score)
    
    max_score = max(scores)
    problems = []
    for i in range(N):
        if scores[i] > max_score:
            problems.append(0)
            continue
        remaining_problems = [A[j] for j in range(M) if S[i][j] == 'x']
        remaining_problems.sort(reverse=True)
        count = 0
        current_score = scores[i]
        for score in remaining_problems:
            current_score += score
            count += 1
            if current_score > max_score:
                problems.append(count)
                break
    
    for p in problems:
        print(p)

solve()