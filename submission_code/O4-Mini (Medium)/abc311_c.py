import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    N = int(data[0])
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        A[i] = int(data[i])
    
    # state: 0 = unvisited, 1 = visiting, 2 = done
    state = [0] * (N + 1)
    # pos[u] will record the index of u in the current path list when state[u] == 1
    pos = [0] * (N + 1)
    
    for start in range(1, N + 1):
        if state[start] != 0:
            continue
        path = []
        u = start
        # walk until we hit a visited node
        while state[u] == 0:
            state[u] = 1
            pos[u] = len(path)
            path.append(u)
            u = A[u]
        # if we hit a node that's in the current path, we found a cycle
        if state[u] == 1:
            cycle_start = pos[u]
            cycle = path[cycle_start:]
            print(len(cycle))
            print(" ".join(map(str, cycle)))
            return
        # otherwise we hit a node already fully processed (state == 2), so mark this path as done
        for v in path:
            state[v] = 2

if __name__ == "__main__":
    main()