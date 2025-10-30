import sys
from collections import deque

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    ptr = 0
    N, M, K = int(input[ptr]), int(input[ptr+1]), int(input[ptr+2])
    ptr +=3
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b = int(input[ptr]), int(input[ptr+1])
        adj[a].append(b)
        adj[b].append(a)
        ptr +=2
    guards = []
    for _ in range(K):
        p, h = int(input[ptr]), int(input[ptr+1])
        guards.append( (p, h) )
        ptr +=2
    max_remaining = [-1]*(N+1)
    q = deque()
    for p, h in guards:
        if h > max_remaining[p]:
            max_remaining[p] = h
            q.append( (p, h) )
    while q:
        u, r = q.popleft()
        if r ==0:
            continue
        for v in adj[u]:
            if r-1 > max_remaining[v]:
                max_remaining[v] = r-1
                q.append( (v, r-1) )
    guarded = [i for i in range(1, N+1) if max_remaining[i] >=0]
    print(len(guarded))
    print(' '.join(map(str, guarded)))

if __name__ == "__main__":
    main()