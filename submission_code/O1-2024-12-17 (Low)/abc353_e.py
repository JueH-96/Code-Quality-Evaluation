def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1:]
    
    # To handle up to 3e5 total characters, we'll build a trie where each node
    # has a "count" (how many strings pass through) and up to 26 children.
    # Then the sum of all LCP's among all pairs is simply
    #   sum_{node at depth >= 1} ( count[node] choose 2 ).
    #
    # Because each pair belonging to a node at depth d indicates an additional
    # character of common prefix for that pair.
    
    # A helper for combinations of 2.
    def c2(x):
        return x*(x-1)//2
    
    # We'll store:
    #   trie[node][ch] = index of the child node (or -1 if none)
    #   count[node] = how many strings go through this node.
    # "node" will be an integer index in [0..trie_size-1].
    
    # The maximum number of nodes is at most sum of lengths of all strings + 1 (for root).
    total_len = sum(len(s) for s in S)
    # Build arrays (lists). Using a list for children. Each node has 26 slots.
    trie = [[-1]*26 for _ in range(total_len + 1)]
    count = [0]*(total_len + 1)
    
    # Insertion routine
    next_node_idx = 1  # 0 is root
    for word in S:
        cur = 0
        count[cur] += 1
        for ch in word:
            c_index = ord(ch) - ord('a')
            if trie[cur][c_index] == -1:
                trie[cur][c_index] = next_node_idx
                next_node_idx += 1
            cur = trie[cur][c_index]
            count[cur] += 1
    
    # We'll DFS from the root to accumulate the sum of choose2(count[node]) for depth >= 1.
    sys.setrecursionlimit(10**7)
    ans = 0
    
    stack = [(0, 0)]  # (node, depth)
    while stack:
        node, depth = stack.pop()
        # from depth>=1, each pair passing through this node contributes +1 to LCP
        if depth > 0:
            ans += c2(count[node])
        for c in range(26):
            child = trie[node][c]
            if child != -1:
                stack.append((child, depth+1))
    
    print(ans)

# Do not forget to call main()
main()