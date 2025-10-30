class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # We want to find, for each query, the string in wordsContainer having the:
        #   1) Longest common suffix with the query.
        #   2) If there is a tie, choose the container string with the smallest length.
        #   3) If there is still a tie, choose the one that appeared earliest in wordsContainer.
        #
        # We can solve this efficiently by building a trie of the reversed words in wordsContainer.
        # Each node in the trie will keep track of the "best" index (in wordsContainer) of any
        # string that passes through that node. "Best" is determined by the rules above.
        #
        # Then, for each query, we reverse it and walk down the trie as far as possible. The node
        # at which we can no longer continue contains information (its "best" index) about the
        # string that has the longest possible suffix match. We use that index as our answer.

        # To implement the "best" tracking at each node:
        # - Each node stores (best_index, best_length).
        # - Whenever we insert a new reversed word w of length L at index 'idx',
        #   if the node has no best_index yet, we set best_index=idx, best_length=L.
        #   If the node has an existing (bst_idx, bst_len),
        #     we compare L vs bst_len:
        #       if L < bst_len, we replace with (idx, L)
        #       if L == bst_len but idx < bst_idx, we replace with (idx, L)
        #     otherwise we keep the old one.
        #
        # For queries, if we can follow the reversed query's characters, we keep going.
        # If we cannot follow at some character, we stop. The node at which we stop has
        # the "best" index for the largest suffix matched so far. If we never move from root
        # (i.e., the first character doesn't match anything), we use the root's best info.
        #
        # Building the trie takes O(sum of lengths of wordsContainer).
        # Querying each reversed query takes time proportional to the length of the query,
        # so total is O(sum of lengths of wordsQuery). This should be efficient enough
        # under the given constraints.

        import sys
        sys.setrecursionlimit(10**7)

        class TrieNode:
            __slots__ = ['children', 'best_index', 'best_len']  # to save memory
            def __init__(self):
                # children[c] will be the next node corresponding to character c
                # c in [0..25] for 'a'..'z'
                self.children = [None]*26
                self.best_index = -1  # index of the best word
                self.best_len = float('inf')  # length of that best word

        def update_best(node: TrieNode, idx: int, length: int):
            # Update node's (best_index, best_len) with a new (idx, length) if needed
            if node.best_index == -1:
                node.best_index = idx
                node.best_len = length
            else:
                if length < node.best_len:
                    node.best_index = idx
                    node.best_len = length
                elif length == node.best_len and idx < node.best_index:
                    node.best_index = idx
                    node.best_len = length

        # Build trie from reversed wordsContainer
        root = TrieNode()

        def char_to_index(c):
            return ord(c) - ord('a')

        # Insert reversed word into trie
        for i, word in enumerate(wordsContainer):
            length = len(word)
            cur = root
            # Update root best info
            update_best(cur, i, length)

            rw = word[::-1]  # reversed word
            for ch in rw:
                idx_c = char_to_index(ch)
                if cur.children[idx_c] is None:
                    cur.children[idx_c] = TrieNode()
                cur = cur.children[idx_c]
                # update best info in the node we go to
                update_best(cur, i, length)

        ans = []
        # Query each reversed wordQuery
        for q in wordsQuery:
            cur = root
            # We'll track how far we can go in the trie
            # but even if we can't go far, the best_index at the node
            # we stop at (including root at the beginning) is the correct
            # answer for the largest matched suffix.
            best_idx = cur.best_index

            rq = q[::-1]
            for ch in rq:
                idx_c = char_to_index(ch)
                if cur.children[idx_c] is None:
                    # can't go deeper
                    break
                cur = cur.children[idx_c]
                best_idx = cur.best_index

            ans.append(best_idx)

        return ans