class TrieNode:
    __slots__ = ['children', 'count']
    def __init__(self):
        self.children = {}
        self.count = 0

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        if len(words) < k:
            return [0] * len(words)
        
        root = TrieNode()
        
        # Insert all words into the trie, counting all prefixes
        for word in words:
            node = root
            for c in word:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
                node.count += 1
        
        answer = []
        
        for word in words:
            # Temporarily remove the current word's prefixes
            node = root
            for c in word:
                node = node.children[c]
                node.count -= 1
            
            # Now find the longest prefix with count >= k
            max_len = 0
            node = root
            while True:
                found = False
                for c, child in node.children.items():
                    if child.count >= k:
                        max_len += 1
                        node = child
                        found = True
                        break
                if not found:
                    break
            
            answer.append(max_len)
            
            # Restore the counts for the current word
            node = root
            for c in word:
                node = node.children[c]
                node.count += 1
        
        return answer