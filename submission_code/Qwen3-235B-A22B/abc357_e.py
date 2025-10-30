import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    
    visited = [False] * (N + 1)
    cycle_length = [0] * (N + 1)
    distance = [0] * (N + 1)
    
    for u in range(1, N+1):
        if not visited[u]:
            path = []
            pos = dict()
            current = u
            while True:
                if visited[current]:
                    break
                visited[current] = True
                path.append(current)
                pos[current] = len(path) - 1
                current = a[current - 1]
            
            if current in pos:
                # Found a cycle
                start = pos[current]
                cl = len(path) - start
                # Assign cycle nodes
                for i in range(start, len(path)):
                    node = path[i]
                    cycle_length[node] = cl
                    distance[node] = 0
                # Assign non-cycle nodes (before the cycle)
                for i in range(start):
                    node = path[i]
                    cycle_length[node] = cl
                    distance[node] = start - i
            else:
                # The path leads to a previously processed node
                v = current
                cl = cycle_length[v]
                dv = distance[v]
                for i in range(len(path)):
                    node = path[i]
                    cycle_length[node] = cl
                    distance[node] = dv + (len(path) - i)
    
    total = 0
    for i in range(1, N+1):
        total += distance[i] + cycle_length[i]
    print(total)

if __name__ == "__main__":
    main()