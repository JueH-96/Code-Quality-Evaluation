class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # We will build a trie of the reversed words from wordsContainer.
        # Each node in the trie will maintain a "best" candidate indicating
        # which word (from wordsContainer) provides the optimal answer for the
        # path (reversed suffix) reaching that node.
        #
        # "best" is stored as a tuple (length_of_word, index_of_word).
        # By storing length first, we easily compare smaller lengths first; if
        # there's a tie in length, we compare the index to keep the earliest word.
        #
        # Then for each query, we reverse it and walk down the trie as far as we can.
        # The "best" tuple at the last node we can reach is our answer. If we cannot
        # go further at any point, we use the node's current "best".

        class TrieNode:
            __slots__ = ('children', 'best')  # To save memory
            def __init__(self):
                # children: dict  {char -> TrieNode}
                self.children = {}
                # best: (length, index)
                self.best = None

        def update_best(current_best, new_candidate):
            # current_best and new_candidate are tuples of the form (length, index).
            # Return the better one according to:
            #  - smaller length is better
            #  - if same length, smaller index is better
            if current_best is None:
                return new_candidate
            if new_candidate[0] < current_best[0]:
                return new_candidate
            if new_candidate[0] == current_best[0] and new_candidate[1] < current_best[1]:
                return new_candidate
            return current_best

        # Build the trie of reversed wordsContainer
        root = TrieNode()
        for i, w in enumerate(wordsContainer):
            node = root
            # Insert the reversed word into the trie
            rev_w = w[::-1]
            candidate = (len(w), i)
            # Update the root's best
            node.best = update_best(node.best, candidate)
            for ch in rev_w:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
                node.best = update_best(node.best, candidate)

        # Now answer each query by traversing the trie with reversed query
        ans = []
        for q in wordsQuery:
            node = root
            best_tuple = node.best  # The root's best, in case we cannot move deeper
            for ch in reversed(q):
                if ch not in node.children:
                    # Can't go further in the trie, so break
                    break
                node = node.children[ch]
                best_tuple = node.best
                # Continue until we can or we run out of characters
            # The best word index is stored at best_tuple[1]
            ans.append(best_tuple[1])

        return ans