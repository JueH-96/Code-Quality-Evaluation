from typing import List

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        # Edge case: if after removal there are fewer than k strings for every i, answers are all 0
        if k == n:
            return [0] * n

        # Define a trie node
        class Node:
            __slots__ = ('children','count','depth','is_B','word_indices')
            def __init__(self, depth: int):
                self.children = {}         # char -> Node
                self.count = 0             # how many words pass here
                self.depth = depth         # length of this prefix
                self.is_B = False          # mark if this is a Type B node (count == k and depth > depthA)
                self.word_indices = None   # list of word indices passing here (only for Type B)

        # Build the trie and counts
        root = Node(0)
        for idx, w in enumerate(words):
            node = root
            node.count += 1
            for ch in w:
                if ch not in node.children:
                    node.children[ch] = Node(node.depth + 1)
                node = node.children[ch]
                node.count += 1

        # Gather all nodes in a list
        nodes = []
        stack = [root]
        while stack:
            nd = stack.pop()
            nodes.append(nd)
            for child in nd.children.values():
                stack.append(child)

        # Find depthA_max (Type A prefixes) and collect all nodes with count == k
        depthA = 0
        all_B = []
        for nd in nodes:
            if nd.count >= k + 1:
                if nd.depth > depthA:
                    depthA = nd.depth
            if nd.count == k:
                all_B.append(nd)

        # From those count==k nodes pick only those with depth > depthA (Type B that can improve beyond depthA)
        B_nodes = []
        for nd in all_B:
            if nd.depth > depthA:
                nd.is_B = True
                nd.word_indices = []
                B_nodes.append(nd)

        # For each word, walk its path and record its index in any Type B node on that path
        for idx, w in enumerate(words):
            node = root
            for ch in w:
                node = node.children[ch]
                if node.is_B:
                    node.word_indices.append(idx)

        # Sort Type B nodes by descending depth
        B_nodes.sort(key=lambda nd: nd.depth, reverse=True)

        # Initialize answers to depthA and track which indices still need improvement
        ans = [depthA] * n
        remaining = set(range(n))

        # Process each Type B node in descending depth:
        # for all i not in its subtree, we can set ans[i] = node.depth (if not already set higher)
        for nd in B_nodes:
            if not remaining:
                break
            d = nd.depth
            S = set(nd.word_indices)  # the k indices in this subtree
            # iterate over a snapshot of remaining
            for i in list(remaining):
                if i not in S:
                    ans[i] = d
                    remaining.remove(i)

        return ans