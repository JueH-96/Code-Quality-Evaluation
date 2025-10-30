import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    visited = [False] * (N + 1)

    for start in range(1, N + 1):
        if not visited[start]:
            path = []
            path_indices = dict()
            current = start
            while True:
                if visited[current]:
                    if current in path_indices:
                        idx = path_indices[current]
                        cycle = path[idx:]
                        print(len(cycle))
                        print(' '.join(map(str, cycle)))
                        return
                    else:
                        for v in path:
                            visited[v] = True
                        break
                else:
                    visited[current] = True
                    path_indices[current] = len(path)
                    path.append(current)
                    current = A[current - 1]

if __name__ == "__main__":
    main()