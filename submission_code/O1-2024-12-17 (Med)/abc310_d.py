def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    input_data = sys.stdin.read().strip().split()
    N, T, M = map(int, input_data[:3])
    pairs = input_data[3:]
    
    # If T == 1 and there's at least one incompatible pair, answer is 0 immediately.
    # If T == 1 and M == 0, answer is 1 (all in one team).
    # We'll handle that as a quick check:
    if T == 1:
        if M == 0:
            print(1)
        else:
            print(0)
        return
    
    # Build adjacency bitmasks (0-based indexing)
    # adjacencyBit[i] will be a bitmask of nodes incompatible with i
    adjacencyBit = [0]*N
    idx = 0
    for _ in range(M):
        a = int(pairs[idx]) - 1
        b = int(pairs[idx+1]) - 1
        idx += 2
        adjacencyBit[a] |= (1 << b)
        adjacencyBit[b] |= (1 << a)
    
    # Precompute degrees to order nodes by descending degree (helps prune faster)
    degrees = [bin(adjacencyBit[i]).count("1") for i in range(N)]
    # Create an ordering of nodes by descending adjacency
    orderedNodes = sorted(range(N), key=lambda x: degrees[x], reverse=True)
    # posOf[node] = position of that node in the ordering
    posOf = [0]*N
    for i, nd in enumerate(orderedNodes):
        posOf[nd] = i
    
    # We'll store colors in color[node], where node is the actual ID (0..N-1).
    # We'll color them in the order given by orderedNodes.
    color = [-1]*N
    
    # Precompute popcount for all bitmasks up to 2^T - though T <= N <= 10
    # but the mask is for colors used, so up to 2^T
    MAX_MASK = 1 << T
    popcount = [0]*MAX_MASK
    for m in range(1, MAX_MASK):
        popcount[m] = popcount[m >> 1] + (m & 1)
    
    answer = [0]  # use list to mutate in inner function
    
    def backtrack(pos, usedColorsMask):
        if pos == N:
            # All nodes colored; check if we used exactly T distinct colors
            if popcount[usedColorsMask] == T:
                answer[0] += 1
            return
        
        # Pruning: if even giving each remaining node a new color wouldn't reach T
        usedCount = popcount[usedColorsMask]
        if usedCount + (N - pos) < T:
            return
        
        node = orderedNodes[pos]
        
        # If we've already used T distinct colors,
        # then we can only color with those T.
        # Otherwise, we can choose from all T colors.
        if usedCount == T:
            color_choices = []
            # collect colors that are in usedColorsMask
            mask_copy = usedColorsMask
            c = 0
            while mask_copy > 0:
                if (mask_copy & 1) == 1:
                    color_choices.append(c)
                mask_copy >>= 1
                c += 1
        else:
            color_choices = range(T)
        
        for c in color_choices:
            # Check adjacency conflict with already colored nodes
            conflict = False
            # We only need to check among the pos nodes colored so far
            # i.e. j in [0..pos-1]
            for j in range(pos):
                other_node = orderedNodes[j]
                if color[other_node] == c:
                    # If there's an edge between node and other_node, conflict
                    if (adjacencyBit[node] & (1 << other_node)) != 0:
                        conflict = True
                        break
            if conflict:
                continue
            
            old_color = color[node]
            color[node] = c
            newUsedColorsMask = usedColorsMask | (1 << c)
            backtrack(pos+1, newUsedColorsMask)
            color[node] = old_color
    
    # Quick edge case: if T > N, no valid way (each team must have at least one player)
    if T > N:
        print(0)
        return
    
    backtrack(0, 0)
    print(answer[0])

# Don't forget to call main()
if __name__ == "__main__":
    main()