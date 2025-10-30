import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    parents = list(map(int, sys.stdin.readline().split()))
    
    parent = [0] * (N + 1)
    for i in range(2, N + 1):
        parent[i] = parents[i - 2]
    
    depth = [0] * (N + 1)
    for i in range(2, N + 1):
        depth[i] = depth[parent[i]] + 1
    
    import sys as system
    max_y = [ -system.maxsize ] * (N + 1)
    for _ in range(M):
        x, y = map(int, sys.stdin.readline().split())
        if y > max_y[x]:
            max_y[x] = y
    
    current_max = [0] * (N + 1)
    current_max[1] = depth[1] + max_y[1]
    for i in range(2, N + 1):
        candidate = depth[i] + max_y[i]
        current_max[i] = max(candidate, current_max[parent[i]])
    
    count = 0
    for v in range(1, N + 1):
        if current_max[v] >= depth[v]:
            count += 1
    print(count)

if __name__ == "__main__":
    main()