class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        from math import inf
        
        # We'll construct a trie over the reversed container words.
        # Each trie node holds:
        #   - a dictionary of children keyed by character
        #   - the index of the "best" word among those that pass through this node (best_idx)
        #   - the length of that best word (best_len)
        #
        # "Best" word is determined by:
        #  1) smallest length
        #  2) if tie in length, smallest index
        #
        # During queries, we'll reverse the query string and walk the trie as far as we can.
        # The node where we stop (either through matching all characters or failing mid-way)
        # determines the best matched word index by looking at best_idx there.

        trie = []         # list of dicts for children: trie[node][char] = next_node
        best_idx = []     # best word index at each node
        best_len = []     # length of best word at each node

        def new_node():
            trie.append({})
            best_idx.append(-1)
            best_len.append(inf)
            return len(trie) - 1

        def update_best(node: int, w_index: int, w_length: int):
            # Update the node's best word if the new word is better by the problem's criteria
            if w_length < best_len[node] or (w_length == best_len[node] and w_index < best_idx[node]):
                best_len[node] = w_length
                best_idx[node] = w_index

        # Create the root of the trie
        root = new_node()

        # Build the trie with reversed container words
        for i, word in enumerate(wordsContainer):
            rev_word = word[::-1]
            # Update the root node for the zero-length match
            update_best(root, i, len(word))
            cur = root
            # Insert each character
            for c in rev_word:
                if c not in trie[cur]:
                    nxt = new_node()
                    trie[cur][c] = nxt
                else:
                    nxt = trie[cur][c]
                cur = nxt
                update_best(cur, i, len(word))

        # Process each query
        ans = []
        for q in wordsQuery:
            rev_q = q[::-1]
            cur = root
            best_node = root  # Track the furthest match
            for c in rev_q:
                if c not in trie[cur]:
                    break
                cur = trie[cur][c]
                best_node = cur
            ans.append(best_idx[best_node])

        return ans