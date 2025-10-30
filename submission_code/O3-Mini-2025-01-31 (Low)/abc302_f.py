def main():
    import sys
    from collections import deque

    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    # Read all sets and build mapping from element to a list of set indices.
    sets = []
    elem_to_sets = [[] for _ in range(m + 1)]
    for i in range(n):
        a = int(next(it))
        current = []
        for _ in range(a):
            num = int(next(it))
            current.append(num)
            elem_to_sets[num].append(i)
        sets.append(current)
    
    # If any one set already contains both 1 and m, answer is 0.
    for i in range(n):
        has1 = False
        has_m = False
        for num in sets[i]:
            if num == 1:
                has1 = True
            if num == m:
                has_m = True
            if has1 and has_m:
                print(0)
                return
    
    # We now want to combine some sets with common elements to eventually 
    # obtain a set that contains both 1 and m.
    # The allowed operation is merging two sets if they share a common element.
    # If we select a connected collection (using intersections of sets) which
    # eventually gives a union over all chosen sets that has both 1 and m,
    # the minimum number of operations required equals (number_of_sets_used - 1).
    #
    # We can construct a graph where each node corresponds to a set.
    # Two sets are connected if they share at least one element.
    # But rather than explicitly constructing that graph (which might be expensive),
    # we use a multi-source BFS from all sets that contain 1. 
    # We then propagate via common elements (processing each element only once).
    # When we first hit a set that contains m, the BFS distance (in terms of number
    # of nodes, sets used) minus 1 is our answer.
    
    dist = [-1] * n
    used_elem = [False] * (m + 1)
    q = deque()

    # Start from all sets that contain 1.
    for idx in elem_to_sets[1]:
        if dist[idx] == -1:
            dist[idx] = 1
            q.append(idx)
    
    ans = -1
    while q:
        cur = q.popleft()

        # Check if the union (current set) contains m.
        # Since operations merge sets, reaching any set containing m means the union
        # of sets on our BFS path has {1, m}
        if m in sets[cur]:
            ans = dist[cur] - 1
            break

        # For every element in the current set, use it to get adjacent sets.
        for e in sets[cur]:
            if not used_elem[e]:
                used_elem[e] = True
                for nxt in elem_to_sets[e]:
                    if dist[nxt] == -1:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
    
    print(ans)

if __name__ == '__main__':
    main()