import sys

def find_cycle():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    A = [0] + A  # Convert to 1-based indexing
    visited = [False] * (N + 1)

    for i in range(1, N + 1):
        if not visited[i]:
            path = []
            in_path = set()
            current = i
            while True:
                if visited[current]:
                    if current in in_path:
                        # Found a cycle
                        idx = path.index(current)
                        cycle = path[idx:]
                        if len(cycle) >= 2:
                            print(len(cycle))
                            print(' '.join(map(str, cycle)))
                            return
                    break
                else:
                    visited[current] = True
                    path.append(current)
                    in_path.add(current)
                    current = A[current]

find_cycle()