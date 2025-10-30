def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Read inputs
    try:
        N = int(data[0])
    except:
        return
    if len(data) < 3:
        return
    S = data[1]
    T = data[2]

    # Step 1. Build the letter‐mapping from S to T.
    # For each position i, the letter S[i] must always map to the same letter T[i].
    mapping = {}
    for i in range(N):
        a = S[i]
        b = T[i]
        if a in mapping:
            if mapping[a] != b:
                # Inconsistent mapping; impossible to achieve T
                sys.stdout.write("-1")
                return
        else:
            mapping[a] = b

    # Step 2. Count the number of letters that need to be changed.
    # We only “need an operation” on a letter x if mapping[x] != x.
    op_count = 0
    for ch in mapping:
        if mapping[ch] != ch:
            op_count += 1

    # Step 3. Detect cycles.
    # Why are cycles a problem? In one operation you can replace all occurrences of x by y.
    # If you have a chain like a->b, b->c, you can process the chain in order and one operation per edge suffices.
    # But if you have a cycle (for example, a->b and b->a) then there is no “starting point” – in fact,
    # you must break the cycle by using an auxiliary letter.
    # This extra auxiliary replacement adds one extra operation per cycle.
    #
    # Note: When S does not contain all 26 letters, you can always choose an auxiliary letter outside S.
    # However, if S uses all 26 letters and the non‐identity mappings form a cycle that covers all the letters involved,
    # then no letter is available as an auxiliary and the transformation is impossible.
    #
    # In our approach we view the mapping (for letters c in S with mapping[c]!=c) as a function graph
    # on at most 26 nodes. Components which form a cycle will require one extra replacement.
    
    visited = set()
    cycle_count = 0
    # Only consider letters for which a change is needed (i.e. mapping[ch] != ch)
    for ch in mapping:
        if mapping[ch] == ch or ch in visited:
            continue
        current_path_index = {}
        current_path = []
        cycle_found = False
        cycle_size = 0
        cur = ch
        while True:
            if cur in current_path_index:
                # Found a cycle–the nodes in current_path from current_path_index[cur] onward form the cycle.
                cycle_found = True
                cycle_size = len(current_path) - current_path_index[cur]
                break
            if cur in visited:
                break
            current_path_index[cur] = len(current_path)
            current_path.append(cur)
            nxt = mapping.get(cur, None)
            # If there is no next letter or if the mapping is trivial (nxt == cur) then stop.
            if nxt is None or nxt == cur:
                break
            # Also, if nxt is present but fixed (i.e. mapping[nxt] == nxt) then we treat it as terminal.
            if nxt not in mapping or mapping.get(nxt) == nxt:
                break
            cur = nxt
        # Mark all nodes in the current DFS branch as visited.
        for node in current_path:
            visited.add(node)
        if cycle_found:
            # To break a cycle you must use an auxiliary letter that is NOT in the cycle.
            # There are 26 lowercase letters in total.
            # If the cycle uses all 26 letters (i.e. cycle_size == 26) then no spare letter is available.
            if cycle_size >= 26:
                sys.stdout.write("-1")
                return
            cycle_count += 1

    # The total minimum operations is the number of non‐identity mappings (one per edge)
    # plus one extra per cycle (to break the cycle).
    result = op_count + cycle_count
    sys.stdout.write(str(result))


if __name__ == '__main__':
    main()