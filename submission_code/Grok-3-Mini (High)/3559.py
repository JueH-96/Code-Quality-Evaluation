from collections import deque
from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build Trie
        root = {}
        for word in words:
            node = root
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
        
        # BFS
        length = len(target)
        dist = [-1] * (length + 1)
        dist[0] = 0
        queue = deque([0])
        
        while queue:
            pos = queue.popleft()
            curr_dist = dist[pos]
            
            # Generate all possible jumps from pos
            node_cur = root
            for len_add in range(1, length - pos + 1):
                idx_char = pos + len_add - 1
                char = target[idx_char]
                if char not in node_cur:
                    break
                node_cur = node_cur[char]
                j = pos + len_add
                if dist[j] == -1:
                    dist[j] = curr_dist + 1
                    queue.append(j)
        
        return dist[length] if dist[length] != -1 else -1