def solve():
    import sys
    from collections import defaultdict, deque

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    idx = 2

    # Read all sets
    sets_list = []
    contains_1 = []  # list of indices of sets that contain 1
    contains_M = []  # list of indices of sets that contain M

    # We'll also store, for each element e, which sets contain e
    element_to_sets = defaultdict(list)
    
    for i in range(N):
        A_i = int(input_data[idx])
        idx += 1
        s = list(map(int, input_data[idx:idx + A_i]))
        idx += A_i
        sets_list.append(s)
    
    # Identify sets containing 1 and/or M and build reverse mapping
    for i, s in enumerate(sets_list):
        # Turn it into a set just once for membership checks
        st = set(s)
        if 1 in st:
            contains_1.append(i)
        if M in st:
            contains_M.append(i)
        for val in s:
            element_to_sets[val].append(i)

    # If any set already contains both 1 and M, answer is 0
    for i, s in enumerate(sets_list):
        st = set(s)
        if 1 in st and M in st:
            print(0)
            return

    # If there are no sets containing 1 or no sets containing M, impossible
    if not contains_1 or not contains_M:
        print(-1)
        return

    # Build adjacency: for each element e, the sets that contain e will form edges
    # We'll only build "chain" edges to keep adjacency smaller
    adjacency = [[] for _ in range(N)]
    for e, sets_with_e in element_to_sets.items():
        # If k sets contain e, connect them in a chain
        # (like a star or chain is enough for them to be in the same component)
        sets_with_e.sort()  # not really necessary to sort, but no harm
        for j in range(len(sets_with_e) - 1):
            u = sets_with_e[j]
            v = sets_with_e[j + 1]
            adjacency[u].append(v)
            adjacency[v].append(u)

    # Multi-source BFS from all sets that contain 1
    # Once we reach a set that contains M, we have our answer
    dist = [-1] * N
    queue = deque()
    for start in contains_1:
        dist[start] = 0
        queue.append(start)

    # Sets that contain M will be potential end points
    target_sets = set(contains_M)

    while queue:
        cur = queue.popleft()
        # If this set contains M, we're done
        if cur in target_sets:
            print(dist[cur])
            return
        for nxt in adjacency[cur]:
            if dist[nxt] == -1:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)

    # If we finished BFS without reaching any set containing M, it's impossible
    print(-1)

def main():
    solve()

if __name__ == "__main__":
    main()