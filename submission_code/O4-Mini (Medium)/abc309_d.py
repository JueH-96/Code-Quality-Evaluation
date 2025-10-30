import sys
import threading
from collections import deque

def main():
    import sys
    input = sys.stdin.readline

    N1, N2, M = map(int, input().split())
    # Build adjacency lists for the two sides separately
    adjA = [[] for _ in range(N1+1)]
    adjB = [[] for _ in range(N2+1)]
    
    for _ in range(M):
        a, b = map(int, input().split())
        # both in A
        if b <= N1:
            adjA[a].append(b)
            adjA[b].append(a)
        # both in B
        elif a > N1:
            # shift indices by N1
            u = a - N1
            v = b - N1
            adjB[u].append(v)
            adjB[v].append(u)
        # else it's a cross‐edge which we ignore for BFS

    # BFS on A from node 1
    distA = [-1] * (N1+1)
    dq = deque([1])
    distA[1] = 0
    while dq:
        u = dq.popleft()
        for w in adjA[u]:
            if distA[w] == -1:
                distA[w] = distA[u] + 1
                dq.append(w)
    maxA = max(distA[1:])  # furthest distance in A

    # BFS on B from node N2 (which corresponds to original N1+N2)
    distB = [-1] * (N2+1)
    startB = N2
    distB[startB] = 0
    dq = deque([startB])
    while dq:
        u = dq.popleft()
        for w in adjB[u]:
            if distB[w] == -1:
                distB[w] = distB[u] + 1
                dq.append(w)
    maxB = max(distB[1:])  # furthest distance in B

    # The best we can do is pick the furthest in A and the furthest in B
    # The distance is distA + 1 (new edge) + distB
    answer = maxA + 1 + maxB
    print(answer)

if __name__ == "__main__":
    main()