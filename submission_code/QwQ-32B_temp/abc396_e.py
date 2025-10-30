import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        X, Y, Z = map(int, sys.stdin.readline().split())
        adj[X].append((Y, Z))
        adj[Y].append((X, Z))
    
    c = [None] * (N + 1)
    
    for node in range(1, N + 1):
        if c[node] is None:
            c[node] = 0
            q = deque()
            q.append(node)
            while q:
                current = q.popleft()
                for (neighbor, z) in adj[current]:
                    if c[neighbor] is None:
                        c[neighbor] = c[current] ^ z
                        q.append(neighbor)
                    else:
                        if (c[current] ^ c[neighbor]) != z:
                            print(-1)
                            return
    
    # Compute bit counts
    counts = [0] * 31
    for v in range(1, N + 1):
        num = c[v]
        for bit in range(31):
            if (num >> bit) & 1:
                counts[bit] += 1
    
    x = 0
    for bit in range(31):
        cnt = counts[bit]
        zeros = N - cnt
        if zeros < cnt:
            x |= (1 << bit)
    
    A = [0] * (N + 1)
    for v in range(1, N + 1):
        A[v] = x ^ c[v]
    
    print(' '.join(map(str, A[1:N+1])))

if __name__ == "__main__":
    main()