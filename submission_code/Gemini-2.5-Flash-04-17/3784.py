from typing import List
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0  # Number of words in the original list having this prefix
        self.depth = 0
        self.word_end_indices = []
        self.max_d_k = 0  # Max depth of node with count >= k in subtree (including self)
        self.max_d_k_plus_1 = 0  # Max depth of node with count >= k+1 in subtree (including self)
        # Store children max_k temporarily for pre-order pass sibling calculation
        self._children_max_k_list = None

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)

        # Handle the case where removing an element makes it impossible to select k strings
        # This happens only when k > n-1. Since 1 <= k <= n, this means k = n.
        if k > n - 1:
             # If k = n, removing any word leaves n-1 words.
             # We need to pick k=n strings from n-1 words, which is impossible.
             # The problem statement says if removing leaves < k strings, answer is 0.
             return [0] * n

        # Build Trie
        root = TrieNode()
        
        for i in range(n):
            curr = root
            for char in words[i]:
                if char not in curr.children:
                    curr.children[char] = TrieNode()
                    curr.children[char].depth = curr.depth + 1
                curr = curr.children[char]
            curr.word_end_indices.append(i) # Mark end of word

        # Post-order DFS to calculate counts (words passing through) and max_d_k, max_d_k_plus_1 in subtree
        def compute_all_info(node):
            # Count words passing through this node = words ending here + words passing through children
            node.count = len(node.word_end_indices) 
            
            # Initialize max_d based on current node's *own* depth if its count meets threshold
            # Note: Root has depth 0. If root.count >= k, max_d_k from node itself is 0.
            # Max depth >= 1 nodes are generally what we care about for positive LCP lengths.
            # Initializing with node.depth is correct as it includes the possibility of the prefix itself.
            node.max_d_k = node.depth if node.count >= k else 0 
            node.max_d_k_plus_1 = node.depth if node.count >= k + 1 else 0 

            children_max_k_tuples = [] # To calculate sibling max_k for pre-order pass

            # Iterate children in a predictable order (e.g., sorted keys)
            sorted_child_chars = sorted(node.children.keys())
            for char in sorted_child_chars:
                child = node.children[char]
                # child_count is the number of original words passing through child node
                child_count = compute_all_info(child) 
                node.count += child_count # Add words passing through children to parent's count

                # Update max_d based on children's subtrees
                node.max_d_k = max(node.max_d_k, child.max_d_k)
                node.max_d_k_plus_1 = max(node.max_d_k_plus_1, child.max_d_k_plus_1)
                
                children_max_k_tuples.append((char, child.max_d_k)) # Collect children max_k

            node._children_max_k_list = children_max_k_tuples # Store for pre-order pass

            return node.count # Return count passing through this node

        # Start post-order DFS from root. The total count of root will be n.
        # The max_d_k_plus_1 calculated at the root will be the global max depth >= k+1.
        compute_all_info(root) 
        
        # Global max depth of a node with count >= k+1.
        # This value is available for *any* removed index i.
        # If root has count >= k+1 (i.e., n >= k+1), its max_d_k_plus_1 includes 0.
        # We care about positive LCP lengths, but max(..., 0) handles this implicitly.
        D_k_plus_1_global = root.max_d_k_plus_1

        answer = [0] * n

        # Pre-order DFS to calculate answers
        # max_k_outside_path: max depth of count-k node NOT in the subtree of current node
        # This is equivalent to max depth of count-k node NOT on the path from root to current node (excluding current node itself)
        def calculate_answers(node, max_k_outside_path):
            
            # For words ending at this node (indices in node.word_end_indices),
            # the path is from root to this node.
            # Max depth of count-k node *not* on this path is max_k_outside_path.
            # The answer for index i is max( D_k_plus_1 (globally), max depth of count-k node NOT on path for words[i] ).
            # max depth of count-k node NOT on path for words[i] = max_k_outside_path (passed parameter)
            for i in node.word_end_indices:
                 answer[i] = max(D_k_plus_1_global, max_k_outside_path)

            # Calculate max_sibling_k_depth for each child to pass down
            children_max_k_values = [d for _, d in node._children_max_k_list]
            
            # Precompute prefix and suffix max for children_max_k_values
            m = len(children_max_k_values)
            prefix_max = [0] * m
            suffix_max = [0] * m
            if m > 0:
                prefix_max[0] = children_max_k_values[0]
                for i in range(1, m):
                    prefix_max[i] = max(prefix_max[i-1], children_max_k_values[i])
                suffix_max[m-1] = children_max_k_values[m-1]
                for i in range(m-2, -1, -1):
                    suffix_max[i] = max(suffix_max[i+1], children_max_k_values[i])

            # Recurse on children
            for child_idx, (char, _) in enumerate(node._children_max_k_list):
                child = node.children[char]
                
                max_sibling_k_depth = 0
                if child_idx > 0:
                    max_sibling_k_depth = max(max_sibling_k_depth, prefix_max[child_idx-1])
                if child_idx < m - 1:
                    max_sibling_k_depth = max(max_sibling_k_depth, suffix_max[child_idx+1])
                
                # When going to child, the max_k_outside_path for the child is the max of:
                # 1. max_k_outside_path for the current node (anything outside the current node's subtree)
                # 2. max_depth_k_in_sibling_subtrees (max k-depth among siblings' subtrees)
                new_max_k_outside_path = max(max_k_outside_path, max_sibling_k_depth)
                
                calculate_answers(child, new_max_k_outside_path)

        calculate_answers(root, 0) # Start DFS from root, initial max_k_outside_path is 0

        return answer