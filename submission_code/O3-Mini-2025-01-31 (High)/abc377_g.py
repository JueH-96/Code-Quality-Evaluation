def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0])
    
    # We create a trie structure where each node stores two pieces of information:
    #   - children: a dictionary mapping a character to the next Trie node.
    #   - min_val: the minimum value among all strings that pass through this node
    #       where for a string S (of length L) inserted and reaching a node at depth d,
    #       the associated value is: (L - 2*d)
    # The reason for this is that if we want to transform a string T into some S,
    # using only delete-from-end (cost=1 per delete) and add-letter (cost=1 per add),
    # the minimum cost is achieved by
    #  • deleting characters until a common prefix of length d is reached and then
    #  • adding the remaining (L - d) letters.
    # So the total cost is: |T| + L - 2*d.
    # When comparing to the option of making T empty (which costs |T|),
    # it suffices to choose the best extra term among prior S_i’s:
    #   extra = (L_j - 2*d)  where L_j = len(S_j) and d is the lcp(S_k, S_j).
    # And because turning T into empty is always available (cost = |T|),
    # we compare against candidate 0.
    
    class TrieNode:
        __slots__ = ('children', 'min_val')
        def __init__(self):
            self.children = {}
            self.min_val = float('inf')
    
    root = TrieNode()
    out_lines = []
    # Process each string in order.
    # For k-th string (S_k), we want the minimum total cost
    #   cost = len(S_k) + min(0, min_{j<k}{len(S_j) - 2 * lcp(S_k, S_j)})
    # We query the trie (which contains all S_1,...,S_{k-1})
    # by following the path that S_k would take. At depth d (1-indexed),
    # any string S_j that has S_k[0:d] contributes candidate = len(S_j) - 2*d.
    # We take the minimum candidate among the nodes along the matching path.
    
    for i in range(n):
        s = data[i+1].rstrip("
")
        best = 0   # The option to delete all letters in T (empty string) costs |T| which is len(s) + 0.
        cur = root
        # Walk down along s as long as matching nodes exist.
        # d is the length of the prefix matched.
        for d, ch in enumerate(s, start=1):
            if ch in cur.children:
                cur = cur.children[ch]
                if cur.min_val < best:
                    best = cur.min_val
            else:
                break
        # The answer for this S_k is: len(s) + (best candidate)
        ans = len(s) + best
        out_lines.append(str(ans))
        
        # Now, insert s into the trie so that future queries (for k'>k) can find it.
        # For the root (depth 0) we use candidate = len(s) - 2*0 = len(s)
        cur = root
        cand = len(s)
        if cand < cur.min_val:
            cur.min_val = cand
        for d, ch in enumerate(s, start=1):
            # For a node at depth d, the candidate value from s is:
            #    len(s) - 2*d 
            cand = len(s) - 2*d
            if ch not in cur.children:
                cur.children[ch] = TrieNode()
                cur = cur.children[ch]
                cur.min_val = cand
            else:
                cur = cur.children[ch]
                if cand < cur.min_val:
                    cur.min_val = cand

    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()