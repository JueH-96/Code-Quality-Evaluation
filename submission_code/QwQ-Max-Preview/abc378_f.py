import sys
from sys import stdin

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(stdin.readline())
    adj = [[] for _ in range(N+1)]  # nodes are 1-based
    for _ in range(N-1):
        u, v = map(int, stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    
    degrees = [0]*(N+1)
    for u in range(1, N+1):
        degrees[u] = len(adj[u])
    
    S = [u for u in range(1, N+1) if degrees[u] == 2]
    s_set = set(S)
    answer = 0
    
    for u in S:
        for neighbor in adj[u]:
            current = neighbor
            prev = u
            found = False
            while True:
                if current in s_set:
                    answer += 1
                    found = True
                    break
                if degrees[current] != 3:
                    break
                # Find next node (not prev)
                next_node = None
                for v in adj[current]:
                    if v != prev:
                        next_node = v
                        break
                if next_node is None:
                    break
                prev = current
                current = next_node
            if found:
                break  # no need to continue this direction
    
    # Each pair is counted twice
    print(answer // 2)

if __name__ == '__main__':
    main()