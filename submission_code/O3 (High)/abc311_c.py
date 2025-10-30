import sys

def main() -> None:
    # Read every integer from stdin at once (fast and robust against line breaks)
    data = list(map(int, sys.stdin.buffer.read().split()))
    N = data[0]
    A = [0] + data[1:]          # 1-indexed: edge i -> A[i]

    visited = [False] * (N + 1) # already processed in previous searches

    for start in range(1, N + 1):
        if visited[start]:
            continue

        path_index = {}  # vertex -> position in current path
        order = []       # current path in order of appearance
        v = start

        while True:
            if v in path_index:                 # cycle discovered!
                cycle = order[path_index[v]:]   # extract the cycle vertices
                print(len(cycle))
                print(' '.join(map(str, cycle)))
                return

            if visited[v]:                      # reached a node handled before
                break

            # mark and advance
            visited[v] = True
            path_index[v] = len(order)
            order.append(v)
            v = A[v]                            # follow the unique outgoing edge

    # According to the problem statement this line is never reached
    # (there is always at least one cycle).
    # It is left only for completeness.
    raise RuntimeError("No cycle found, although one should exist.")

if __name__ == "__main__":
    main()