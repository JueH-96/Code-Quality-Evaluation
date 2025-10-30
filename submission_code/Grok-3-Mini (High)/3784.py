import collections
from typing import List

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        if n <= k:
            return [0] * n
        
        # Define TrieNode
        class TrieNode:
            def __init__(self):
                self.children = {}  # char to TrieNode
                self.count = 0
        
        # Build trie
        root = TrieNode()
        for word in words:
            node = root
            node.count += 1  # increment count at root (depth 0)
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.count += 1  # increment count at each node along the path
        
        # Compute word_prefix_counts for each word
        word_prefix_counts = [None] * n
        for i in range(n):
            counts = []
            current_node = root
            counts.append(current_node.count)  # depth 0
            for char in words[i]:
                current_node = current_node.children[char]
                counts.append(current_node.count)  # append count at each depth
            word_prefix_counts[i] = counts  # list of counts from depth 0 to len(word)
        
        # BFS to get nodes per depth
        nodes_at_depth = collections.defaultdict(list)
        queue = collections.deque()
        queue.append((root, 0))  # (node, depth)
        while queue:
            curr_node, depth = queue.popleft()
            nodes_at_depth[depth].append(curr_node)
            for child in curr_node.children.values():
                queue.append((child, depth + 1))
        
        # Find max_L
        max_L = max(nodes_at_depth.keys())
        
        # Compute num_high for each depth
        num_high = {}
        for depth in nodes_at_depth:
            count_ge_k = sum(1 for node in nodes_at_depth[depth] if node.count >= k)
            num_high[depth] = count_ge_k
        
        # Define check function
        def check(L, idx):
            word_len = len(words[idx])
            if L > word_len:
                return num_high.get(L, 0) > 0
            else:
                c_w = word_prefix_counts[idx][L]
                num_high_L = num_high.get(L, 0)
                if c_w >= k + 1:
                    return True
                elif c_w == k:
                    return num_high_L > 1
                else:  # c_w < k
                    return num_high_L > 0
        
        # For each index, binary search to find the maximum L
        answer = []
        for i in range(n):
            low = 0
            high = max_L
            res = -1
            while low <= high:
                mid = (low + high) // 2
                if check(mid, i):
                    res = mid
                    low = mid + 1
                else:
                    high = mid - 1
            answer.append(res)
        
        return answer