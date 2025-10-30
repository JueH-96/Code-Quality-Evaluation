class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        class TrieNode:
            __slots__ = ['children']
            def __init__(self):
                self.children = {}
        
        root = TrieNode()
        
        # Build the trie with all prefixes of the words
        for word in words:
            node = root
            for c in word:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
        
        n = len(target)
        pos = 0
        steps = 0
        
        while pos < n:
            current_node = root
            max_l = 0
            # Traverse the trie to find the maximum possible length starting at pos
            for i in range(pos, n):
                c = target[i]
                if c in current_node.children:
                    current_node = current_node.children[c]
                    max_l += 1
                else:
                    break  # No further extension possible here
            if max_l == 0:
                return -1  # No valid prefix found starting at current position
            pos += max_l
            steps += 1
        
        return steps