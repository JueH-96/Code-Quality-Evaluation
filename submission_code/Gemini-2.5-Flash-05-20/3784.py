import collections

class TrieNode:
    def __init__(self, depth, node_id):
        self.children = {}
        self.count = 0  # Number of words in the original 'words' array passing through this node
        self.depth = depth
        self.node_id = node_id
        
        # For segment tree range updates
        # Smallest/Largest original index of a word passing through this node
        self.first_word_idx = float('inf')  
        self.last_word_idx = float('-inf')  
        
        # If this node marks the end of a word, stores its original index
        self.is_word_end_for_idx = -1 

class SegmentTree:
    def __init__(self, n, initial_value=0):
        self.n = n
        # tree stores the maximum value in the segment
        self.tree = [0] * (4 * n) 
        # lazy stores the max value to propagate downwards
        self.lazy = [0] * (4 * n) 
        self._build(0, 0, n - 1, initial_value)

    def _build(self, node_idx, start, end, initial_value):
        if start == end:
            self.tree[node_idx] = initial_value
            return
        mid = (start + end) // 2
        self._build(2 * node_idx + 1, start, mid, initial_value)
        self._build(2 * node_idx + 2, mid + 1, end, initial_value)
        self.tree[node_idx] = max(self.tree[2 * node_idx + 1], self.tree[2 * node_idx + 2])

    def _push_down(self, node_idx):
        if self.lazy[node_idx] != 0:
            # Apply lazy value to children
            self.tree[2 * node_idx + 1] = max(self.tree[2 * node_idx + 1], self.lazy[node_idx])
            self.lazy[2 * node_idx + 1] = max(self.lazy[2 * node_idx + 1], self.lazy[node_idx])
            self.tree[2 * node_idx + 2] = max(self.tree[2 * node_idx + 2], self.lazy[node_idx])
            self.lazy[2 * node_idx + 2] = max(self.lazy[2 * node_idx + 2], self.lazy[node_idx])
            # Clear lazy value for current node
            self.lazy[node_idx] = 0

    def range_max_update(self, l, r, val):
        self._range_max_update(0, 0, self.n - 1, l, r, val)

    def _range_max_update(self, node_idx, start, end, l, r, val):
        # Current segment completely outside query range
        if r < start or end < l:
            return
        
        # Current segment is completely within query range
        if l <= start and end <= r: 
            self.tree[node_idx] = max(self.tree[node_idx], val)
            self.lazy[node_idx] = max(self.lazy[node_idx], val)
            return

        self._push_down(node_idx) # Push down lazy value before going deeper

        mid = (start + end) // 2
        self._range_max_update(2 * node_idx + 1, start, mid, l, r, val)
        self._range_max_update(2 * node_idx + 2, mid + 1, end, l, r, val)
        self.tree[node_idx] = max(self.tree[2 * node_idx + 1], self.tree[2 * node_idx + 2])

    def query(self, l, r):
        return self._query(0, 0, self.n - 1, l, r)

    def _query(self, node_idx, start, end, l, r):
        # Current segment completely outside query range
        if r < start or end < l:
            return 0 

        # Current segment is completely within query range
        if l <= start and end <= r:
            return self.tree[node_idx]

        self._push_down(node_idx) # Push down lazy value before querying children

        mid = (start + end) // 2
        p1 = self._query(2 * node_idx + 1, start, mid, l, r)
        p2 = self._query(2 * node_idx + 2, mid + 1, end, l, r)
        return max(p1, p2)


class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        N = len(words)
        
        # 1. Trie Construction & Initial Counts
        root = TrieNode(0, 0)
        all_trie_nodes = [root]
        node_id_counter = 0

        for i, word in enumerate(words):
            curr = root
            for char in word:
                if char not in curr.children:
                    node_id_counter += 1
                    new_node = TrieNode(curr.depth + 1, node_id_counter)
                    curr.children[char] = new_node
                    all_trie_nodes.append(new_node)
                curr = curr.children[char]
                curr.count += 1
            curr.is_word_end_for_idx = i # Mark the original index of the word ending here

        # 2. DFS on Trie for Word Index Ranges & base_max_lcp
        # base_max_lcp captures max depth for nodes with count >= k + 1
        base_max_lcp = 0 
        
        # Post-order DFS to calculate first_word_idx and last_word_idx
        def dfs_populate_indices_and_base_lcp(node: TrieNode):
            nonlocal base_max_lcp
            
            # If this node represents an end of a word, update its range with its own index
            if node.is_word_end_for_idx != -1:
                node.first_word_idx = node.is_word_end_for_idx
                node.last_word_idx = node.is_word_end_for_idx

            # Recursively call for children
            for child in node.children.values():
                dfs_populate_indices_and_base_lcp(child)
                # Aggregate min/max indices from children
                node.first_word_idx = min(node.first_word_idx, child.first_word_idx)
                node.last_word_idx = max(node.last_word_idx, child.last_word_idx)
            
            # Update base_max_lcp: for nodes that will always have count >= k
            if node.count >= k + 1:
                base_max_lcp = max(base_max_lcp, node.depth)

        dfs_populate_indices_and_base_lcp(root)
        
        # 3. Collect EqKNodes (nodes with count == k)
        eq_k_nodes = [] # Stores (depth, node_id)
        for node in all_trie_nodes:
            if node.count == k:
                # Ensure it's a valid range (not the default inf/-inf from unvisited/empty children)
                if node.first_word_idx != float('inf'): # Node must actually correspond to words
                    eq_k_nodes.append((node.depth, node.node_id))
        
        # Sort by depth descending to process longer prefixes first
        eq_k_nodes.sort(key=lambda x: x[0], reverse=True)

        # 4. Segment Tree Setup
        # Initialize all answers with base_max_lcp
        seg_tree = SegmentTree(N, base_max_lcp)

        # 5. Process EqKNodes with Segment Tree
        for depth, node_id in eq_k_nodes:
            node = all_trie_nodes[node_id]
            
            # The range of indices where this node's prefix is present
            L = node.first_word_idx
            R = node.last_word_idx

            # Update answers for words NOT in this range [L, R]
            # These are words whose removal does NOT invalidate this node's prefix count.
            
            # Update left part (indices 0 to L-1)
            if L > 0:
                seg_tree.range_max_update(0, L - 1, depth)
            
            # Update right part (indices R+1 to N-1)
            if R < N - 1:
                seg_tree.range_max_update(R + 1, N - 1, depth)
        
        # 6. Final Result
        answer = [0] * N
        for i in range(N):
            answer[i] = seg_tree.query(i, i)
        
        return answer