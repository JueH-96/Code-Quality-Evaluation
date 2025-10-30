class TrieNode:
    __slots__ = ['children', 'best']
    def __init__(self):
        self.children = {}
        self.best = None

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        root = TrieNode()
        for idx, word in enumerate(wordsContainer):
            rword = word[::-1]
            node = root
            if node.best is None:
                node.best = (len(word), idx)
            else:
                curr_len, curr_idx = node.best
                if len(word) < curr_len or (len(word) == curr_len and idx < curr_idx):
                    node.best = (len(word), idx)
            for c in rword:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
                if node.best is None:
                    node.best = (len(word), idx)
                else:
                    curr_len, curr_idx = node.best
                    if len(word) < curr_len or (len(word) == curr_len and idx < curr_idx):
                        node.best = (len(word), idx)
        
        ans = []
        for q in wordsQuery:
            rq = q[::-1]
            node = root
            best_index = node.best[1]
            for c in rq:
                if c in node.children:
                    node = node.children[c]
                    best_index = node.best[1]
                else:
                    break
            ans.append(best_index)
        return ans