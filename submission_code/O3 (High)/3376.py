from typing import List

class Solution:
    class _TrieNode:
        __slots__ = ('children', 'best_idx')
        def __init__(self):
            self.children = {}          # char -> _TrieNode
            self.best_idx = None        # index of the best word that passes this node

    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        """
        For every query string return the index of the word in wordsContainer that
        shares the longest common suffix with the query.  Ties are resolved by
        (1) shorter word length, (2) smaller index.
        """
        # -------- pre–compute lengths (used for fast tie–breaking) ------------
        lens = [len(w) for w in wordsContainer]

        # ----------------------- build reversed trie --------------------------
        root = self._TrieNode()

        def _maybe_update(node: 'Solution._TrieNode', cand_idx: int):
            """
            Put cand_idx into node.best_idx if it is better w.r.t. the
            tie–breaking rule.
            """
            if node.best_idx is None:
                node.best_idx = cand_idx
                return
            best = node.best_idx
            if lens[cand_idx] < lens[best] or (lens[cand_idx] == lens[best] and cand_idx < best):
                node.best_idx = cand_idx

        for idx, word in enumerate(wordsContainer):
            # update root
            _maybe_update(root, idx)

            node = root
            for ch in reversed(word):          # turning suffix into prefix
                nxt = node.children.get(ch)
                if nxt is None:
                    nxt = self._TrieNode()
                    node.children[ch] = nxt
                node = nxt
                _maybe_update(node, idx)

        # ------------------------- answer queries -----------------------------
        ans = []
        for q in wordsQuery:
            node = root
            best_idx = node.best_idx          # empty suffix always matches

            for ch in reversed(q):
                nxt = node.children.get(ch)
                if nxt is None:               # path breaks – cannot go deeper
                    break
                node = nxt
                best_idx = node.best_idx      # deeper => longer common suffix

            ans.append(best_idx)

        return ans