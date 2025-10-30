def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0])
    s_list = data[1:]
    
    # We will use a trie. Each node stores:
    # - children: dict mapping letter -> node
    # - min_val: the minimum length among all strings that pass through this node.
    #
    # Why? For any transformation from string X to some string Y (with Y among previously seen strings),
    # the allowed operations (deleting from the end then adding letters) have cost:
    #    cost = (|X| - LCP(X, Y)) + (|Y| - LCP(X, Y))  = |X| + |Y| - 2*LCP(X, Y).
    # To beat the cost of deleting all letters of X (which is cost |X|), we want to find some Y for which
    #   |Y| - 2*LCP(X, Y) is as small as possible.
    #
    # When processing S_k we have X = S_k and the available Y's are the earlier strings.
    # In a trie storing the earlier strings, if we traverse along S_k we can know the depth d (which is LCP length)
    # and at each node we can access the minimum length among the strings sharing that prefix.
    # Therefore, for each achievable prefix length d (starting with d=0 at the root),
    # we have a candidate cost: |Y| - 2*d (with |Y| being the min length among words with that prefix).
    #
    # The answer then becomes:
    #    ans = min( |X|,   |X| + min_{0<=d<=LCP}(min_len(prefix d) - 2*d) )
    # (We always have d=0 available using the root; and note that the cost of converting X to empty is |X|.)
    
    class TrieNode:
        __slots__ = "children", "min_val"
        def __init__(self):
            self.children = {}
            self.min_val = 10**9  # a large number (greater than any possible string length)
    
    # Create the root of the trie.
    root = TrieNode()
    
    output = []
    for idx, s in enumerate(s_list):
        # For k==1 (idx == 0), no previous string exists so the only option is to delete all of s.
        if idx == 0:
            cost = len(s)
            output.append(str(cost))
        else:
            candidate = root.min_val  # for prefix length 0, candidate = (|Y| - 0)
            node = root
            depth = 0
            # Traverse matching prefix of s in the trie.
            for ch in s:
                if ch in node.children:
                    node = node.children[ch]
                    depth += 1
                    # For the current prefix length 'depth', any previous string sharing it
                    # gives candidate value = (min length among those strings) - 2 * depth.
                    tmp = node.min_val - 2 * depth
                    if tmp < candidate:
                        candidate = tmp
                else:
                    break
            # The transformation cost using some previous string Y would be:
            #   cost = |s| + (|Y| - 2*LCP)
            # Also, we can always delete all characters in s, which is cost = |s|.
            cost = len(s) if len(s) < (len(s) + candidate) else (len(s) + candidate)
            output.append(str(cost))
        
        # Insert s into the trie, updating min_val along the path.
        node = root
        if len(s) < node.min_val:
            node.min_val = len(s)
        for ch in s:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            if len(s) < node.min_val:
                node.min_val = len(s)
    
    sys.stdout.write("
".join(output))

if __name__ == '__main__':
    main()