from collections import deque

def solve(N, K):
    if N % 2 == 0 and K % 2 == 0:
        return "No"
    
    points = deque(range(N))
    black = 0
    
    while points:
        if len(points) == 1:
            return "Yes"
        
        if len(points) % 2 == 0:
            points.rotate(-K)
            black += 1
        else:
            points.rotate(K)
            black += 1
        
        if black == N:
            return "Yes"
        
        points.pop()
    
    return "No"

T = int(input())
for _ in range(T):
    N, K = map(int, input().split())
    print(solve(N, K))