import sys
from collections import deque

def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    # We'll zero-index elements: 1..M -> 0..M-1
    sets = [None] * N
    sets_by_elem = [[] for _ in range(M)]
    for i in range(N):
        Ai = int(input().strip())
        arr = list(map(int, input().split()))
        # convert to 0-based elements
        arr0 = [x-1 for x in arr]
        sets[i] = arr0
        for e0 in arr0:
            sets_by_elem[e0].append(i)

    # starting sets: those containing element 1 => e0 = 0
    start_sets = sets_by_elem[0]
    # target sets: those containing element M => e0 = M-1
    target_e0 = M - 1
    if target_e0 < 0 or target_e0 >= M:
        # No such element M in zero-based range
        print(-1)
        return
    target_sets = sets_by_elem[target_e0]

    # If no start or no target, impossible
    if not start_sets or not target_sets:
        print(-1)
        return

    # Quick check: any set contains both 1 and M -> answer 0
    # We'll mark target sets for fast lookup
    is_target = [False] * N
    for s in target_sets:
        is_target[s] = True

    for s in start_sets:
        if is_target[s]:
            print(0)
            return

    # We create a bipartite graph in BFS:
    # - set-nodes: ids 0..N-1
    # - elem-nodes: ids N..N+M-1  (for element e0 use id = N + e0)
    total_nodes = N + M
    dist = [-1] * total_nodes
    dq = deque()

    # initialize BFS from all start_sets (they have distance 0 merges)
    for s in start_sets:
        dist[s] = 0
        dq.append(s)

    # BFS
    while dq:
        node = dq.popleft()
        if node < N:
            # this is a set-node
            s = node
            # traverse to all its element-nodes
            for e0 in sets[s]:
                eid = N + e0
                if dist[eid] == -1:
                    # moving set->elem does not increase merge count
                    dist[eid] = dist[s]
                    dq.append(eid)
        else:
            # this is an element-node
            e0 = node - N
            # traverse to all set-nodes containing this element
            for s2 in sets_by_elem[e0]:
                if dist[s2] == -1:
                    # moving elem->set increases merge count by 1
                    dist[s2] = dist[node] + 1
                    # if this set is a target, we reached a set that contains M
                    if is_target[s2]:
                        print(dist[s2])
                        return
                    dq.append(s2)

    # if BFS completes without finding a target set, impossible
    print(-1)

if __name__ == "__main__":
    main()