def main():
    import sys
    from collections import deque
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))

    # Read each set and build a mapping from element -> list of set indices that contain it.
    sets = []
    element_to_sets = [[] for _ in range(m + 1)]
    for i in range(n):
        cnt = int(next(it))
        current_set = []
        for _ in range(cnt):
            num = int(next(it))
            current_set.append(num)
            # Add set index i for the element num.
            element_to_sets[num].append(i)
        sets.append(current_set)

    # If any one set already contains both 1 and m, we need 0 operations.
    for s in sets:
        has1 = False
        hasM = False
        for num in s:
            if num == 1:
                has1 = True
            elif num == m:
                hasM = True
            if has1 and hasM:
                print(0)
                return

    # We can merge sets if they share any common element.
    # We form a "graph" where each node corresponds to a set, and an edge exists between two sets if they share a common element.
    # When we merge k sets together (in a chain), we perform k-1 operations.
    # So, if we find a chain (path) of sets connecting one that contains 1 with one that contains m,
    # the minimum number of merge operations is the number of edges (which is chain length - 1).

    # BFS over sets: nodes are set indices.
    # Start from all sets which contain 1.
    dist = [-1] * n  # distance in the "merge graph" (number of merges performed)
    dq = deque()
    for i in element_to_sets[1]:
        if dist[i] == -1:
            dist[i] = 0
            dq.append(i)

    # To avoid processing the same element over and over, we mark when an element has been "processed" (its neighbor sets have been added)
    processed_elem = [False] * (m + 1)

    while dq:
        cur_set = dq.popleft()
        # For each element in the current set
        for num in sets[cur_set]:
            if processed_elem[num]:
                continue  # already processed this element
            # For each set adjacent via this common element
            for nxt_set in element_to_sets[num]:
                if dist[nxt_set] == -1:
                    dist[nxt_set] = dist[cur_set] + 1
                    # Check if this next set contains m.
                    for x in sets[nxt_set]:
                        if x == m:
                            # Found a chain connecting a set with 1 and a set with m.
                            # The number of operations is the distance (number of merges needed).
                            print(dist[nxt_set])
                            return
                    dq.append(nxt_set)
            processed_elem[num] = True

    # If nothing reached a set containing m, print -1.
    print(-1)

if __name__ == '__main__':
    main()