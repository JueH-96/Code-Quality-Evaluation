import collections
from typing import List

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        
        # If removing an element leaves fewer than k strings, the answer is 0.
        # This occurs if n - 1 < k. Given the constraint k <= n, this simplifies to n == k.
        if n == k:
            return [0] * n

        class TrieNode:
            def __init__(self):
                self.children = collections.defaultdict(TrieNode)
                self.count = 0
                # Max depth of a node in this subtree (inclusive) with count >= k
                self.max_depth_ge_k_in = 0
                # For sibling max calculation
                self.child_list = []
                self.child_map = {}
                self.child_prefix_max = []
                self.child_suffix_max = []

        # 1. Build Trie and count frequencies
        root = TrieNode()
        for word in words:
            curr = root
            curr.count += 1
            for char in word:
                curr = curr.children[char]
                curr.count += 1
        
        # 2. Get all nodes and their depths using BFS (pre-order traversal)
        all_nodes_with_depth = []
        q = collections.deque([(root, 0)])
        visited_ids = {id(root)}
        
        while q:
            node, depth = q.popleft()
            all_nodes_with_depth.append((node, depth))
            
            for child in node.children.values():
                if id(child) not in visited_ids:
                    visited_ids.add(id(child))
                    q.append((child, depth + 1))
    
        # 3. Post-order traversal (by iterating backwards) to compute max_depth_ge_k_in
        for node, depth in reversed(all_nodes_with_depth):
            max_val = depth if node.count >= k else 0
            for child in node.children.values():
                max_val = max(max_val, child.max_depth_ge_k_in)
            node.max_depth_ge_k_in = max_val

        # 4. Pre-order traversal to compute sibling maxes
        for node, depth in all_nodes_with_depth:
            # Sort children by char to have a consistent order
            sorted_children = sorted(node.children.items())
            node.child_list = [child for char, child in sorted_children]
            node.child_map = {child: i for i, (char, child) in enumerate(sorted_children)}
            
            m = len(node.child_list)
            if m == 0:
                continue
            
            node.child_prefix_max = [0] * m
            node.child_prefix_max[0] = node.child_list[0].max_depth_ge_k_in
            for i in range(1, m):
                node.child_prefix_max[i] = max(node.child_prefix_max[i-1], node.child_list[i].max_depth_ge_k_in)
                
            node.child_suffix_max = [0] * m
            node.child_suffix_max[m-1] = node.child_list[m-1].max_depth_ge_k_in
            for i in range(m-2, -1, -1):
                node.child_suffix_max[i] = max(node.child_suffix_max[i+1], node.child_list[i].max_depth_ge_k_in)

        # 5. Compute the final answer for each word
        answers = [0] * n
        for i in range(n):
            word = words[i]
            
            path_nodes = [root]
            curr = root
            for char in word:
                curr = curr.children[char]
                path_nodes.append(curr)
            
            # L_on_path: Max depth of a node on the path with count > k
            max_on = 0
            for depth, node in enumerate(path_nodes):
                if node.count > k:
                    max_on = max(max_on, depth)
                    
            # L_off_path: Max depth of a node with count >= k NOT on the path
            max_off = 0
            for depth in range(len(word)):
                parent = path_nodes[depth]
                child = path_nodes[depth+1]
                
                child_idx = parent.child_map.get(child)
                if child_idx is None: continue

                m = len(parent.child_list)
                max_sibling_val = 0
                if child_idx > 0:
                    max_sibling_val = max(max_sibling_val, parent.child_prefix_max[child_idx-1])
                if child_idx < m - 1:
                    max_sibling_val = max(max_sibling_val, parent.child_suffix_max[child_idx+1])
                
                max_off = max(max_off, max_sibling_val)
                
            answers[i] = max(max_on, max_off)
            
        return answers