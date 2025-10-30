# YOUR CODE HERE
import sys
import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    N,*rest = map(int, sys.stdin.read().split())
    a = rest[:N]
    a = [x-1 for x in a]
    visited = [False]*N
    reachable_count = [0]*N
    for u in range(N):
        if not visited[u]:
            path = []
            indices = {}
            v = u
            while True:
                if visited[v]:
                    if reachable_count[v] > 0:
                        add = reachable_count[v]
                        for node in reversed(path):
                            add +=1
                            reachable_count[node] = add
                    break
                if v in indices:
                    idx = indices[v]
                    cycle_nodes = path[idx:]
                    cycle_size = len(cycle_nodes)
                    for node in cycle_nodes:
                        reachable_count[node] = cycle_size
                    add = cycle_size
                    for node in reversed(path[:idx]):
                        add +=1
                        reachable_count[node] = add
                    break
                indices[v] = len(path)
                path.append(v)
                v = a[v]
            for node in path:
                visited[node] = True
    print(sum(reachable_count))

if __name__ == "__main__":
    main()