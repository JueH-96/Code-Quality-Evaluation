import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, X = map(int, input().split())
    A = [0] + list(map(int, input().split()))
    B = [0] + list(map(int, input().split()))
    P = [0] + list(map(int, input().split()))
    Q = [0] + list(map(int, input().split()))

    # Build reverse graphs for P and Q
    revP = [[] for _ in range(N+1)]
    revQ = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        revP[P[i]].append(i)
        revQ[Q[i]].append(i)

    # BFS from X on revP to find which nodes can reach X via P
    from collections import deque
    visR = [False] * (N+1)
    dq = deque([X])
    visR[X] = True
    while dq:
        u = dq.popleft()
        for v in revP[u]:
            if not visR[v]:
                visR[v] = True
                dq.append(v)

    # BFS from X on revQ to find which nodes can reach X via Q
    visB = [False] * (N+1)
    dq = deque([X])
    visB[X] = True
    while dq:
        u = dq.popleft()
        for v in revQ[u]:
            if not visB[v]:
                visB[v] = True
                dq.append(v)

    # Check if any red ball is in unreachable node
    for i in range(1, N+1):
        if A[i] and not visR[i]:
            print(-1)
            return
    # Check blue balls
    for i in range(1, N+1):
        if B[i] and not visB[i]:
            print(-1)
            return

    # markR[i] = whether we need to operate at i for red paths
    # markB similarly for blue
    markR = [False] * (N+1)
    markB = [False] * (N+1)

    # For each red ball start, walk up P until X, marking nodes
    for i in range(1, N+1):
        if A[i]:
            curr = i
            while curr != X and not markR[curr]:
                markR[curr] = True
                curr = P[curr]

    # For each blue ball start, walk up Q until X, marking nodes
    for i in range(1, N+1):
        if B[i]:
            curr = i
            while curr != X and not markB[curr]:
                markB[curr] = True
                curr = Q[curr]

    # Count the union of marked nodes, excluding X
    ans = 0
    for i in range(1, N+1):
        if i == X:
            continue
        if markR[i] or markB[i]:
            ans += 1

    print(ans)

if __name__ == "__main__":
    main()