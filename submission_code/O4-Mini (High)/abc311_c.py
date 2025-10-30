import sys
def main():
    input = sys.stdin.readline
    N = int(input())
    A = [0] + list(map(int, input().split()))
    # state: 0 = unvisited, 1 = in current exploration stack, 2 = fully processed
    state = [0] * (N + 1)
    # pos[v] will store index of v in the current path (only valid when state[v] == 1)
    pos = [-1] * (N + 1)

    for start in range(1, N + 1):
        if state[start] != 0:
            continue
        path = []
        cur = start
        # Walk until we hit a visited node
        while state[cur] == 0:
            state[cur] = 1
            pos[cur] = len(path)
            path.append(cur)
            cur = A[cur]
        # If we land on a node in the current stack, we found a cycle
        if state[cur] == 1:
            idx = pos[cur]
            cycle = path[idx:]
            print(len(cycle))
            print(" ".join(map(str, cycle)))
            return
        # Otherwise we hit a fully processed node (state[cur] == 2), mark this chain as done
        for v in path:
            state[v] = 2

if __name__ == "__main__":
    main()