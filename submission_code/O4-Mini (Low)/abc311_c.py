import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N = int(input())
    A = [0] + list(map(int, input().split()))

    # 0 = unvisited, 1 = visiting, 2 = done
    state = [0] * (N + 1)

    for i in range(1, N + 1):
        if state[i] != 0:
            continue
        path = []
        idx_in_path = dict()
        curr = i

        # walk until we hit a visited node
        while state[curr] == 0:
            state[curr] = 1
            idx_in_path[curr] = len(path)
            path.append(curr)
            curr = A[curr]

        # if we hit a node that's currently on our path, we found a cycle
        if state[curr] == 1:
            start = idx_in_path[curr]
            cycle = path[start:]
            # output result
            print(len(cycle))
            print(" ".join(map(str, cycle)))
            return

        # otherwise, we hit a node already fully processed: mark our path as done
        for v in path:
            state[v] = 2

if __name__ == "__main__":
    main()