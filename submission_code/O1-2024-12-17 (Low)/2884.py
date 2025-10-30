class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        from collections import deque
        
        # Aho-Corasick Trie construction
        # --------------------------------
        # Each node in the trie will have:
        #   edges: dict from character -> next node index
        #   fail: failure link
        #   out: list of lengths of forbidden words that end at this node
        #
        # We'll build the trie of all forbidden patterns, build
        # the fail links, then scan through "word" maintaining
        # the current state in the automaton. Whenever we find
        # a forbidden pattern ending at index i, we update the
        # sliding window start accordingly.

        # Step 1: Build trie structure.
        # We track the trie in arrays/lists because that's typically faster in Python
        # for large tries. However, using a dictionary for edges is also fine.

        # Because each forbidden word has length up to 10, but we may have up to 1e5 such words,
        # the total size of the trie can be up to ~1e6 in the worst case.

        ALPHABET_SIZE = 26  # only lowercase letters
        
        # We'll store edges as a list of dictionaries or lists of size ALPHABET_SIZE
        # for quick indexing 'a' to 'z'.
        # node 0 will be the root.
        trie_edges = []
        fail_link = []
        output = []  # output[i] = list of lengths of patterns that end at node i
        
        def new_node():
            trie_edges.append([-1]*ALPHABET_SIZE)
            fail_link.append(0)
            output.append([])
            return len(trie_edges) - 1
        
        root = new_node()
        
        # Function to map char -> index
        def c2i(c):
            return ord(c) - ord('a')
        
        # Insert all forbidden patterns into the trie
        for pat in forbidden:
            cur = root
            for ch in pat:
                idx = c2i(ch)
                if trie_edges[cur][idx] == -1:
                    trie_edges[cur][idx] = new_node()
                cur = trie_edges[cur][idx]
            # At end of pattern, record its length
            output[cur].append(len(pat))
        
        # Step 2: Build failure links using BFS
        q = deque()
        # For each letter from root, if we have an edge, set fail to 0, otherwise -1 remains
        for c in range(ALPHABET_SIZE):
            nxt = trie_edges[root][c]
            if nxt != -1:
                fail_link[nxt] = root
                q.append(nxt)
            else:
                trie_edges[root][c] = root
        
        # BFS to build fail links
        while q:
            u = q.popleft()
            f = fail_link[u]
            for c in range(ALPHABET_SIZE):
                nxt = trie_edges[u][c]
                if nxt == -1:
                    # If there's no edge from u by char c, link to the fail of the fail state
                    trie_edges[u][c] = trie_edges[f][c]
                else:
                    # If we have an edge, set the fail link and accumulate outputs
                    fail_link[nxt] = trie_edges[f][c]
                    # Merge the output from the fail state
                    output[nxt].extend(output[ trie_edges[f][c] ])
                    q.append(nxt)
        
        # Step 3: Traverse the word and find the maximum valid substring length
        # We'll keep a sliding window [start..i]. If we find a forbidden pattern
        # ending at i, we advance start so that substring no longer contains it.
        
        state = root
        max_len = 0
        start = 0
        
        for i, ch in enumerate(word):
            idx = c2i(ch)
            # Move in trie
            state = trie_edges[state][idx]
            
            # If any pattern ends here, we move start beyond the earliest forbidden start
            if output[state]:
                # Among all matched forbidden patterns, find the largest length
                # Then the earliest beginning is i - length + 1
                # so start must be > that end (i - length + 1 + (0-based)) => +1
                # we pick the largest shift needed
                shift = 0
                for length_p in output[state]:
                    # forbidden pattern is [i - length_p + 1 .. i]
                    begin_index = i - length_p + 1
                    # so valid substring can't include that range
                    shift = max(shift, begin_index + 1)
                # update start if needed
                if shift > start:
                    start = shift
            
            # current substring length
            cur_len = i - start + 1
            if cur_len > max_len:
                max_len = cur_len
        
        return max_len