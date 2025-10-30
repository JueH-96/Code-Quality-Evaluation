def main():
    import sys,sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    # Build the tree: for i>=2, parent is given.
    children = [[] for _ in range(N+1)]
    # Note: person 1 is root.
    # p_2, p_3, ... p_N
    for i in range(2, N+1):
        p = int(next(it))
        children[p].append(i)
    
    # store insurance events per person: each event is (y) (for that person).
    # The insurance bought by person x with parameter y covers x and every descendant v that has:
    #    depth(v) <= depth(x) + y .
    events = [[] for _ in range(N+1)]
    for _ in range(M):
        x = int(next(it))
        y = int(next(it))
        events[x].append(y)
    
    # We are going to do a DFS in the tree.
    # Instead of maintaining a BIT to answer "what is the maximum allowed depth currently in force?"
    # we maintain a global variable curr_max which equals the maximum over all insurances purchased by
    # ancestors in the current DFS path. When we add new events (at the moment of entering a node)
    # we compute:
    #    new_global = max(old_global, max(depth_of_current + y for each event at that node) )
    # Then, the node is covered if and only if curr_max >= node_depth.
    # When we finish processing a node we restore the previous global maximum.
    
    # We'll use an iterative DFS using an explicit stack.
    # Each frame in the DFS will be a tuple:
    #   (v, depth, child_index, saved_global)
    # where saved_global is the parent's curr_max before processing the current node.
    
    ans = 0
    curr_max = -1  # global maximum allowed depth from events on the current DFS path.
    
    stack = []
    # Process the root (person 1) at depth 0.
    d = 0
    saved = curr_max  # before processing root, parent's global is -1.
    local_max = -1
    for y in events[1]:
        local_max = max(local_max, d + y)
    new_global = max(saved, local_max)
    curr_max = new_global
    # A node is covered if curr_max >= its depth.
    if curr_max >= d:
        ans += 1
    # push a frame for the root: (node, depth, next child index, saved_global)
    stack.append((1, d, 0, saved))
    
    # Iterative DFS:
    while stack:
        v, depth_v, child_index, saved_global = stack[-1]
        if child_index < len(children[v]):
            # get the next child to process
            child = children[v][child_index]
            # update the top frame's child_index
            stack[-1] = (v, depth_v, child_index + 1, saved_global)
            child_depth = depth_v + 1
            child_saved = curr_max  # save current global before processing child
            local_child = -1
            for y in events[child]:
                local_child = max(local_child, child_depth + y)
            new_global = max(child_saved, local_child)
            curr_max = new_global
            if curr_max >= child_depth:
                ans += 1
            stack.append((child, child_depth, 0, child_saved))
        else:
            # finished processing children of this node, backtrack:
            # restore the global value that was in force before entering this node.
            curr_max = saved_global
            stack.pop()
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()