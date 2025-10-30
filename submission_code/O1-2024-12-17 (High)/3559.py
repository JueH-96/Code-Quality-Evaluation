class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        from collections import deque
        
        # A simple trie node with 26 possible children and a flag signaling
        # that this node corresponds to the end of a valid prefix
        class TrieNode:
            __slots__ = ('children', 'end')
            def __init__(self):
                self.children = [None] * 26
                self.end = False
        
        # Build the trie. As we insert each word, we mark every prefix as "end"
        # because all prefixes of any word are valid.
        root = TrieNode()
        
        for w in words:
            node = root
            for c in w:
                idx = ord(c) - ord('a')
                if node.children[idx] is None:
                    node.children[idx] = TrieNode()
                node = node.children[idx]
                node.end = True  # Mark each prefix position as valid

        n = len(target)
        dist = [float('inf')] * (n + 1)
        dist[0] = 0
        
        # We'll use BFS over positions in the target, trying all possible valid prefixes
        queue = deque([0])
        
        while queue:
            start_pos = queue.popleft()
            if start_pos == n:
                # If we've reached the end of the target, that's our answer
                return dist[start_pos]
            
            # If we already have a count of strings that would exceed
            # a potential best answer at the end, we might skip, but
            # for simplicity we'll proceed.
            node = root
            # Try matching every possible prefix from start_pos onward
            for j in range(start_pos, n):
                idx = ord(target[j]) - ord('a')
                if node.children[idx] is None:
                    break  # No further match possible
                node = node.children[idx]
                if node.end:
                    # We can form a valid prefix up to j (inclusive)
                    if dist[j + 1] > dist[start_pos] + 1:
                        dist[j + 1] = dist[start_pos] + 1
                        queue.append(j + 1)
        
        return -1 if dist[n] == float('inf') else dist[n]