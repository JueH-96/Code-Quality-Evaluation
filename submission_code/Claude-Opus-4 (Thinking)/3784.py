class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.count = 0
        
        n = len(words)
        answer = []
        
        for i in range(n):
            # If removing the i-th element leaves fewer than k strings
            if n - 1 < k:
                answer.append(0)
                continue
            
            # Build a trie for remaining words
            root = TrieNode()
            root.count = n - 1  # All remaining words have the empty prefix
            
            for j in range(n):
                if j == i:
                    continue
                
                word = words[j]
                node = root
                for char in word:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                    node.count += 1
            
            # Find the deepest node with count >= k
            max_len = 0
            
            def dfs(node, depth):
                nonlocal max_len
                if node.count >= k:
                    max_len = max(max_len, depth)
                for child in node.children.values():
                    dfs(child, depth + 1)
            
            dfs(root, 0)
            answer.append(max_len)
        
        return answer