import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        x = int(input[ptr])
        ptr += 1
        y = int(input[ptr])
        ptr += 1
        z = int(input[ptr])
        ptr += 1
        adj[x].append((y, z))
        adj[y].append((x, z))

    visited = [False] * (N + 1)
    value = [0] * (N + 1)
    components = []

    for i in range(1, N + 1):
        if not visited[i]:
            component = []
            q = deque()
            q.append(i)
            visited[i] = True
            value[i] = 0
            component.append(i)
            valid = True
            while q:
                u = q.popleft()
                for (v, z) in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        value[v] = value[u] ^ z
                        q.append(v)
                        component.append(v)
                    else:
                        if value[v] != (value[u] ^ z):
                            print(-1)
                            return
            components.append(component)

    A = [0] * (N + 1)
    for comp in components:
        bits_count = [0] * 31
        for node in comp:
            val = value[node]
            for bit in range(31):
                if (val >> bit) & 1:
                    bits_count[bit] += 1
        x = 0
        size = len(comp)
        for bit in range(31):
            cnt = bits_count[bit]
            if cnt >= size - cnt:
                x |= (1 << bit)
        for node in comp:
            A[node] = value[node] ^ x

    print(' '.join(map(str, A[1:N+1])))

if __name__ == "__main__":
    main()