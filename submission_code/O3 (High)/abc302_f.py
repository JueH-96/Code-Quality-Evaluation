import sys
from collections import deque

def main() -> None:
    # -------- fast input --------
    data = list(map(int, sys.stdin.buffer.read().split()))
    it = iter(data)
    try:
        N = next(it)
        M = next(it)
    except StopIteration:          # empty input (should not happen with valid tests)
        return

    # containers
    sets_elements = []                  # list of lists – elements of each set
    value_to_sets = [[] for _ in range(M + 1)]   # for each value, the indices of sets containing it
    start_sets = []                     # indices of sets that contain 1
    target_sets = []                    # indices of sets that contain M

    # ---------- read all sets ----------
    for idx in range(N):
        Ai = next(it)
        elems = []
        has1 = hasM = False
        for _ in range(Ai):
            v = next(it)
            elems.append(v)
            value_to_sets[v].append(idx)
            if v == 1:
                has1 = True
            if v == M:
                hasM = True
        sets_elements.append(elems)
        if has1:
            start_sets.append(idx)
        if hasM:
            target_sets.append(idx)

    # if no set contains 1 or M – impossible
    if not start_sets or not target_sets:
        print(-1)
        return

    target_set_ids = set(target_sets)

    # If a start set already contains M, no operation is needed
    if target_set_ids & set(start_sets):
        print(0)
        return

    # ------------- BFS on the implicit intersection graph -------------
    dist = [-1] * N
    q = deque(start_sets)
    for s in start_sets:
        dist[s] = 0

    processed_value = [False] * (M + 1)

    while q:
        s = q.popleft()

        # reached a set that contains M
        if s in target_set_ids:
            print(dist[s])
            return

        nd = dist[s] + 1
        for v in sets_elements[s]:
            if not processed_value[v]:
                processed_value[v] = True
                for t in value_to_sets[v]:
                    if dist[t] == -1:
                        dist[t] = nd
                        q.append(t)

    # unreachable
    print(-1)


if __name__ == "__main__":
    main()