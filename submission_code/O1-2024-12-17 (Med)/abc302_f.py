def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])

    # Read sets
    idx = 2
    sets = []
    contains_1 = [False]*N
    contains_M = [False]*N

    # For each element e in [1..M], store the list of sets that contain e
    # We use M+1 so we can index directly by element e
    e2sets = [[] for _ in range(M+1)]

    for i in range(N):
        A_i = int(input_data[idx])
        idx += 1
        current_set = list(map(int, input_data[idx:idx + A_i]))
        idx += A_i
        sets.append(current_set)
        # Check if 1 or M in current_set
        if 1 in current_set:
            contains_1[i] = True
        if M in current_set:
            contains_M[i] = True
        # Record that set i contains these elements
        for e in current_set:
            e2sets[e].append(i)

    # If any set already contains both 1 and M, answer is 0
    for i in range(N):
        if contains_1[i] and contains_M[i]:
            print(0)
            return

    # Collect all sets that contain 1 (start BFS from them)
    starts = [i for i in range(N) if contains_1[i]]
    if not starts:
        # No set contains 1 => impossible
        print(-1)
        return

    # We'll do a BFS over the "set graph" in an implicit way:
    # dist[i] = minimum number of merges needed to reach set i
    dist = [-1]*N
    queue = deque()
    for s in starts:
        dist[s] = 0
        queue.append(s)

    # We'll also keep track of which elements we've already used to discover neighbors
    used_elem = [False]*(M+1)

    # Standard BFS
    while queue:
        cur_set_idx = queue.popleft()
        d = dist[cur_set_idx]
        # If this set contains M => we've reached a set that has both 1 (implicitly via BFS) and M
        if contains_M[cur_set_idx]:
            print(d)
            return
        # Otherwise, explore neighbors via the elements of cur_set
        for e in sets[cur_set_idx]:
            if not used_elem[e]:
                # For each set that also contains e
                for nxt_set_idx in e2sets[e]:
                    if dist[nxt_set_idx] == -1:
                        dist[nxt_set_idx] = d + 1
                        queue.append(nxt_set_idx)
                # Mark element e as used so we don't repeat adjacency exploration
                used_elem[e] = True

    # If we exit the BFS loop without finding a set containing M, it's impossible
    print(-1)

# Do not forget to call main()
if __name__ == "__main__":
    main()