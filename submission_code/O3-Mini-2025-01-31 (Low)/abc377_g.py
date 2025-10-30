def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0])
    s_list = data[1:]
    
    # We will build a Trie to store the previously seen strings.
    # In each trie node, we store:
    #   children: dictionary mapping letter -> next node
    #   min_len: the minimum length among all strings in the previous set having this prefix.
    # This allows us, for a query string T, to try every prefix length d (starting at the root for d=0)
    # and compute the cost for one candidate U (which gives common prefix length = d) as:
    #      |T| + |U| - 2*d.
    # That is, for the current d, the best candidate using that prefix is determined by the minimal length
    # among strings having that prefix.
    
    class TrieNode:
        __slots__ = ('children', 'min_len')
        def __init__(self):
            self.children = {}
            self.min_len = float('inf')
    
    root = TrieNode()
    output_lines = []
    
    # Process strings in order
    for idx, T in enumerate(s_list):
        # Our goal for string T (S_k) is to find the minimum cost to transform T
        # into either the empty string (cost = len(T)) or any string U from the set
        # S_1, S_2, ..., S_{k-1} (cost = |T| + |U| - 2 * LCP(T, U)).
        best_cost = len(T)  # cost to remove all characters (make empty)
        
        # Query the Trie for previously inserted strings (if any).
        cur = root
        d = 0  # current matched prefix length
        # At the root (prefix of length 0) if any previous string has been inserted,
        # update candidate cost using d=0: cost = len(T) + |U| - 0.
        if cur.min_len != float('inf'):
            candidate = len(T) + cur.min_len  # d = 0 so subtract 0
            if candidate < best_cost:
                best_cost = candidate
        
        # Walk along T and update the candidate as long as possible.
        for ch in T:
            if ch not in cur.children:
                break
            cur = cur.children[ch]
            d += 1
            # For this prefix, if there is any string U in the trie with prefix T[:d],
            # candidate cost = |T| + (|U|) - 2*d.
            if cur.min_len != float('inf'):
                candidate = len(T) + cur.min_len - 2 * d
                if candidate < best_cost:
                    best_cost = candidate
        
        # Append answer for the current string.
        output_lines.append(str(best_cost))
        
        # Now, insert the current string T into the trie.
        cur = root
        # Update the root's min_len: every inserted string qualifies for the empty prefix.
        if len(T) < cur.min_len:
            cur.min_len = len(T)
        for ch in T:
            if ch not in cur.children:
                cur.children[ch] = TrieNode()
            cur = cur.children[ch]
            if len(T) < cur.min_len:
                cur.min_len = len(T)
    
    sys.stdout.write("
".join(output_lines))

if __name__ == '__main__':
    main()