from typing import List
import collections

class Solution:
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.ending_indices = []
            # Store indices of words that *pass through* this node (sorted list)
            self.all_indices = []

        @staticmethod
        def merge_sorted_lists_helper(list1, list2):
            merged = []
            p1, p2 = 0, 0
            while p1 < len(list1) or p2 < len(list2):
                if p1 < len(list1) and (p2 >= len(list2) or list1[p1] <= list2[p2]):
                    merged.append(list1[p1])
                    p1 += 1
                else:
                    merged.append(list2[p2])
                    p2 += 1
            return merged

    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)

        # Although handled by logic, can add explicit check for clarity/minor optimization
        # if n - 1 < k:
        #     return [0] * n

        # Build Trie
        root = self.TrieNode()
        for i, word in enumerate(words):
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = self.TrieNode()
                node = node.children[char]
            node.ending_indices.append(i)

        # Post-order DFS to collect all_indices (sorted list)
        def dfs_collect_indices(node: self.TrieNode):
            all_child_indices = []
            # Iterate through children in deterministic order (e.g., sorted char)
            for char in sorted(node.children.keys()):
                child_node = node.children[char]
                dfs_collect_indices(child_node)
                all_child_indices = self.TrieNode.merge_sorted_lists_helper(all_child_indices, child_node.all_indices)

            # Add indices ending at this node
            node.all_indices = self.TrieNode.merge_sorted_lists_helper(all_child_indices, sorted(node.ending_indices))

        dfs_collect_indices(root)

        answer = [0] * n

        # Pre-order DFS to compute answers
        # max_lcp_from_other_paths: maximum depth L of a node w OUTSIDE the current node's subtree
        # such that |w.all_indices| >= k.
        def dfs_compute_answer(node: self.TrieNode, depth: int, max_lcp_from_other_paths: int):
            I = node.all_indices
            C = len(I)
            L = depth

            # For any index `i` in `I` (passing through the current node),
            # the max LCP derived from nodes *outside* the current subtree
            # where the count condition is met is `max_lcp_from_other_paths`.
            # This is because if `i` is in `I`, it's not in the all_indices list of any node
            # outside the current subtree.
            for i in I:
                 answer[i] = max(answer[i], max_lcp_from_other_paths)

            # Calculate LCP contribution from current node `u` itself if C >= k.
            # This contribution applies to indices *not* in `I`.
            lcp_contrib_from_u = 0
            if C >= k:
                 lcp_contrib_from_u = L

            children_items = sorted(node.children.items())
            num_children = len(children_items)

            if num_children > 0:
                # Calculate max LCP from sibling branches for each child.
                # This is the max depth (if count >= k) among sibling nodes.
                sibling_contrib_values = []
                for char, child_node in children_items:
                     contrib = (depth + 1) if len(child_node.all_indices) >= k else 0
                     sibling_contrib_values.append(contrib)

                # Use prefix/suffix max to find max sibling contrib excluding the current child
                prefix_max_depth = [0] * num_children
                suffix_max_depth = [0] * num_children

                for j in range(num_children):
                    prefix_max_depth[j] = max(prefix_max_depth[j-1], sibling_contrib_values[j]) if j > 0 else sibling_contrib_values[j]

                for j in range(num_children - 1, -1, -1):
                    suffix_max_depth[j] = max(suffix_max_depth[j+1], sibling_contrib_values[j]) if j < num_children - 1 else sibling_contrib_values[j]

                # Recurse on children
                for j in range(num_children):
                    char, child_node = children_items[j]
                    max_from_other_siblings = 0
                    if j > 0:
                        max_from_other_siblings = max(max_from_other_siblings, prefix_max_depth[j-1])
                    if j < num_children - 1:
                        max_from_other_siblings = max(max_from_other_siblings, suffix_max_depth[j+1])

                    # max_lcp_from_other_paths for child v includes:
                    # 1. max_lcp from nodes outside u's subtree (max_lcp_from_other_paths passed to u)
                    # 2. LCP from u itself if C >= k (lcp_contrib_from_u)
                    # 3. max_lcp from v's sibling branches (max_from_other_siblings)
                    next_max_lcp = max(max_lcp_from_other_paths, lcp_contrib_from_u, max_from_other_siblings)
                    dfs_compute_answer(child_node, depth + 1, next_max_lcp)

            # After recursing, update answer based on C-1 >= k.
            # This handles cases where removing `i` (where `i` is IN `I`)
            # allows LCP `L` using `k-1` other strings *within* `I`.
            if C - 1 >= k:
                for i in I:
                    answer[i] = max(answer[i], L)

        # Root node depth is 0. Initial max_lcp_from_other_paths is 0.
        dfs_compute_answer(root, 0, 0)

        return answer