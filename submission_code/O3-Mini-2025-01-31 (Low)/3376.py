from typing import List, Dict

class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        # candidate info as (word_length, index)
        # This candidate will be the best container word (smallest length,
        # and if a tie, smallest index) among all container words that pass
        # through this node.
        self.cand = None  # type: (int, int) or None

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # comparator: smaller word length, then smaller index.
        def update_candidate(old, new):
            if old is None:
                return new
            # old and new are tuples (length, index)
            if new[0] < old[0] or (new[0] == old[0] and new[1] < old[1]):
                return new
            return old

        # Build trie of reversed words from container.
        root = TrieNode()
        # For global candidate for empty suffix, i.e. common suffix ""
        global_cand = None  # tuple: (length, index)
        
        for idx, word in enumerate(wordsContainer):
            word_len = len(word)
            cand_val = (word_len, idx)
            # update global candidate (empty string suffix is common to all)
            global_cand = update_candidate(global_cand, cand_val)
            
            node = root
            # update candidate at root as well
            node.cand = update_candidate(node.cand, cand_val)
            
            # Insert reversed word into trie.
            for ch in word[::-1]:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
                node.cand = update_candidate(node.cand, cand_val)

        # Process each query.
        ans = []
        for query in wordsQuery:
            node = root
            # Start with global candidate (empty suffix) in case no letter matches.
            best_cand = node.cand
            
            for ch in query[::-1]:
                if ch in node.children:
                    node = node.children[ch]
                    best_cand = node.cand
                else:
                    break
            # best_cand is (length, index). We output the index.
            ans.append(best_cand[1])
        return ans