from typing import List
import sys
sys.setrecursionlimit(10**7)

class TrieNode:
    __slots__ = ('children', 'count')
    def __init__(self):
        self.children = {}
        self.count = 0

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        # Build trie and count occurrences
        root = TrieNode()
        # We'll also store for each word the path of trie nodes
        word_nodes = [None] * n
        
        for i, w in enumerate(words):
            node = root
            path = []
            for ch in w:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
                node.count += 1
                path.append(node)
            word_nodes[i] = path
        
        # Compute for each depth d (1-based) how many trie nodes have count >= k
        # Also precompute max_depth
        max_len = max(len(w) for w in words) if words else 0
        nodes_k = [0] * (max_len + 1)  # nodes_k[d] = #nodes at depth d with count>=k
        # Traverse trie to collect counts by depth
        def dfs(node, depth):
            if depth > 0 and node.count >= k:
                nodes_k[depth] += 1
            for child in node.children.values():
                dfs(child, depth + 1)
        dfs(root, 0)
        
        # Precompute the global maximum depth with any nodes_k[d]>0
        base_max = 0
        for d in range(1, max_len + 1):
            if nodes_k[d] > 0:
                base_max = d
        
        # For fast lookup of nodes_k[d]==1
        single_k = [nodes_k[d] == 1 for d in range(max_len + 1)]
        
        ans = [0] * n
        # For each word, determine depths where its path node.count == k
        for i in range(n):
            path = word_nodes[i]
            # bad depths are those depths d where removal would invalidate the only >=k node
            bad = set()
            for d, node in enumerate(path, start=1):
                if node.count == k:
                    bad.add(d)
            # Now find the maximum valid depth after removal
            d = base_max
            while d > 0:
                # if this depth is "bad" and there's exactly one node >=k originally,
                # then after removal none remain
                if d in bad and single_k[d]:
                    d -= 1
                    continue
                # otherwise at least one prefix of length d survives
                break
            ans[i] = d
        
        return ans