import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.buffer.read().split()
    it = iter(data)

    N = int(next(it))
    # 0-based indices
    P = [int(next(it)) - 1 for _ in range(N)]
    A = [int(next(it)) for _ in range(N)]

    visited = [False] * N
    result = [0] * N  # final permutation we will output

    for start in range(N):
        if visited[start]:
            continue

        # collect one cycle in the order given by permutation P
        cycle = []
        cur = start
        while not visited[cur]:
            visited[cur] = True
            cycle.append(cur)
            cur = P[cur]

        k = len(cycle)
        if k == 1:                      # fixed point, cannot change
            result[cycle[0]] = A[cycle[0]]
            continue

        # find the position (in the cycle order) of
        #   - the smallest index inside the cycle
        #   - the smallest value that lives in this cycle
        min_index_in_cycle = cycle[0]
        pos_min_index = 0

        min_value_in_cycle = A[cycle[0]]
        pos_min_value = 0

        for pos in range(1, k):
            idx = cycle[pos]
            if idx < min_index_in_cycle:
                min_index_in_cycle = idx
                pos_min_index = pos
            val = A[idx]
            if val < min_value_in_cycle:
                min_value_in_cycle = val
                pos_min_value = pos

        # number of operations (rotation) that puts the minimal value
        # at the smallest index of the cycle
        shift = (pos_min_value - pos_min_index) % k

        # apply this rotation to the whole cycle
        # after 'shift' operations, value at cycle[pos] comes from
        # cycle[(pos + shift) % k] in the original array
        for pos in range(k):
            from_pos = (pos + shift) % k
            result[cycle[pos]] = A[cycle[from_pos]]

    print(' '.join(map(str, result)))


if __name__ == "__main__":
    main()