# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    S = data[2+M:]
    
    scores = [0] * N
    unsolved = [[] for _ in range(N)]
    
    for i in range(N):
        for j in range(M):
            if S[i][j] == 'o':
                scores[i] += A[j]
            else:
                unsolved[i].append(A[j])
        scores[i] += i + 1
    
    for i in range(N):
        current_score = scores[i]
        needed = 0
        for j in range(N):
            if i != j and current_score <= scores[j]:
                needed += 1
                current_score += max(unsolved[i])
                unsolved[i].remove(max(unsolved[i]))
        print(needed)

solve()