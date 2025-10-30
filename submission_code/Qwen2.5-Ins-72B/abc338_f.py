import sys
from itertools import permutations

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    edges = []
    for i in range(M):
        u = int(data[2 + 3 * i]) - 1
        v = int(data[3 + 3 * i]) - 1
        w = int(data[4 + 3 * i])
        edges.append((u, v, w))
    
    # Create adjacency list
    adj = [[] for _ in range(N)]
    for u, v, w in edges:
        adj[u].append((v, w))
    
    # Function to find the minimum weight of a walk that visits all vertices
    def min_walk_weight():
        min_weight = float('inf')
        for perm in permutations(range(N)):
            weight = 0
            valid = True
            for i in range(N - 1):
                u, v = perm[i], perm[i + 1]
                found = False
                for next_v, w in adj[u]:
                    if next_v == v:
                        weight += w
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if valid:
                min_weight = min(min_weight, weight)
        return min_weight if min_weight != float('inf') else None
    
    result = min_walk_weight()
    if result is None:
        print("No")
    else:
        print(result)

if __name__ == "__main__":
    solve()