import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    S = int(input[ptr]); ptr += 1
    T = int(input[ptr]); ptr += 1

    adj = [[] for _ in range(N + 1)]  # 1-based indexing
    for _ in range(M):
        u = int(input[ptr]); ptr += 1
        v = int(input[ptr]); ptr += 1
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    q = deque()
    initial = (S, T)
    q.append(initial)
    visited.add(initial)

    dist = 0

    while q:
        nq = len(q)
        for _ in range(nq):
            a, b = q.popleft()
            if a == T and b == S:
                print(dist)
                return
            # Move A
            for na in adj[a]:
                if na != b:
                    new_state = (na, b)
                    if new_state not in visited:
                        visited.add(new_state)
                        q.append(new_state)
            # Move B
            for nb in adj[b]:
                if a != nb:
                    new_state = (a, nb)
                    if new_state not in visited:
                        visited.add(new_state)
                        q.append(new_state)
        dist += 1

    print(-1)

if __name__ == "__main__":
    main()