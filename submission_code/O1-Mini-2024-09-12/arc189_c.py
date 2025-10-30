# YOUR CODE HERE
import sys
import sys
import sys
from collections import deque

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N, X = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    P = list(map(int, sys.stdin.readline().split()))
    Q = list(map(int, sys.stdin.readline().split()))
    
    # Build reverse graphs
    red_rev = [[] for _ in range(N+1)]
    blue_rev = [[] for _ in range(N+1)]
    
    for i in range(1, N+1):
        red_rev[P[i-1]].append(i)
        blue_rev[Q[i-1]].append(i)
    
    # BFS for red
    distance_red = [float('inf')] * (N+1)
    q = deque()
    distance_red[X] = 0
    q.append(X)
    while q:
        u = q.popleft()
        for v in red_rev[u]:
            if distance_red[v] == float('inf'):
                distance_red[v] = distance_red[u] +1
                q.append(v)
    
    # BFS for blue
    distance_blue = [float('inf')] * (N+1)
    q = deque()
    distance_blue[X] = 0
    q.append(X)
    while q:
        u = q.popleft()
        for v in blue_rev[u]:
            if distance_blue[v] == float('inf'):
                distance_blue[v] = distance_blue[u] +1
                q.append(v)
    
    max_dist = 0
    possible = True
    for i in range(1, N+1):
        if i == X:
            continue
        if A[i-1]:
            if distance_red[i] == float('inf'):
                possible = False
                break
            max_dist = max(max_dist, distance_red[i])
        if B[i-1]:
            if distance_blue[i] == float('inf'):
                possible = False
                break
            max_dist = max(max_dist, distance_blue[i])
    
    if not possible:
        print(-1)
    else:
        print(max_dist +1 if max_dist !=0 else 0)

if __name__ == "__main__":
    main()