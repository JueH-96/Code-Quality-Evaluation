from collections import defaultdict
from typing import List, Dict, Set, Optional

class TrieNode:
    __slots__ = ['children', 'count', 'is_end_of_word'] # Use slots for memory efficiency

    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.count: int = 0 # Number of words passing through this node
        self.is_end_of_word: bool = False

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        # If after removing one element, fewer than k strings remain, the answer is 0.
        # This happens if the original number of strings n is <= k.
        if n <= k:
             return [0] * n
        
        root = TrieNode()
        nodes_by_depth: Dict[int, Set[TrieNode]] = defaultdict(set)

        # Insert words and collect nodes by depth
        def insert(word: str):
            curr = root
            nodes_by_depth[0].add(curr)
            for i, char in enumerate(word):
                if char not in curr.children:
                    curr.children[char] = TrieNode()
                curr = curr.children[char]
                nodes_by_depth[i + 1].add(curr)
            curr.is_end_of_word = True

        for word in words:
            insert(word)

        # Compute counts using DFS (post-order traversal)
        def compute_count(node: TrieNode) -> int:
            count = 1 if node.is_end_of_word else 0
            for child in node.children.values():
                count += compute_count(child)
            node.count = count
            return count

        compute_count(root) # Start counting from root

        # Get max depth
        max_depth = max(nodes_by_depth.keys(), default=0)

        # Precompute info for each depth
        count_nodes_at_d_ge_k: Dict[int, int] = {}
        single_node_at_d_ge_k: Dict[int, Optional[TrieNode]] = {} # Stores the single node if count is exactly 1

        # Clear nodes_by_depth to potentially save memory after counting is done
        # Convert sets to lists first if needed later, but seems only counts/single nodes are needed now
        # nodes_by_depth = None # Or clear its contents if needed for other parts? No, it's not needed anymore.

        # Re-iterate nodes by depth to populate count_nodes_at_d_ge_k and single_node_at_d_ge_k
        # We need the list representation to iterate potentially multiple nodes at a depth
        nodes_by_depth_list: Dict[int, List[TrieNode]] = {d: list(node_set) for d, node_set in nodes_by_depth.items()}
        nodes_by_depth.clear() # Clear the set version

        for d in range(max_depth + 1):
             nodes_list = nodes_by_depth_list.get(d, [])
             count_ge_k = 0
             unique_node_ge_k = None
             for node in nodes_list:
                 if node.count >= k:
                     count_ge_k += 1
                     # Keep track if we've seen more than one node with count >= k
                     if unique_node_ge_k is None:
                         unique_node_ge_k = node
                     else:
                         unique_node_ge_k = False # Mark as not unique
                         
             count_nodes_at_d_ge_k[d] = count_ge_k
             if count_ge_k == 1:
                 # If count is 1, unique_node_ge_k will hold the single node reference
                 single_node_at_d_ge_k[d] = unique_node_ge_k
             else:
                 single_node_at_d_ge_k[d] = None
                 
        # nodes_by_depth_list is no longer needed
        nodes_by_depth_list.clear()


        # Compute max_d_ge_k and second_max_d_ge_k
        max_d_ge_k = 0
        second_max_d_ge_k = 0
        found_max = False
        # Iterate from max_depth down to 1 (depth 0 is root, always exists, count >= k needs k=1 which is trivial, but question asks for LCP length >= 1)
        for d in range(max_depth, 0, -1):
            if count_nodes_at_d_ge_k.get(d, 0) > 0:
                if not found_max:
                    max_d_ge_k = d
                    found_max = True
                else:
                    second_max_d_ge_k = d
                    break # Found second max, we can stop

        answer = [0] * n
        answer_by_word: Dict[str, int] = {}

        for i, word_to_remove in enumerate(words):
            if word_to_remove in answer_by_word:
                answer[i] = answer_by_word[word_to_remove]
                continue

            L_i = len(word_to_remove)

            # Calculate max_on_path_contrib (max depth d <= L_i where node on path has count >= k+1)
            max_on_path_contrib = 0
            curr = root
            path_nodes: List[Optional[TrieNode]] = [root] # Store nodes on path for quick access
            
            for d in range(L_i):
                char = word_to_remove[d]
                if char not in curr.children:
                     # This should not happen for words from the original input list
                     curr = None 
                     break
                curr = curr.children[char]
                path_nodes.append(curr)

            # path_nodes contains nodes from depth 0 to L_i (if path exists fully)
            # Need to check count >= k+1 at each node on the path
            # The last node in path_nodes is at depth L_i (if word exists)
            if len(path_nodes) == L_i + 1: # Ensure the full path was found
                 for d in range(len(path_nodes)): # Iterate depths 0 to L_i
                      node = path_nodes[d]
                      if node is not None and node.count >= k + 1:
                           max_on_path_contrib = max(max_on_path_contrib, d)

            # Calculate max_not_on_path_contrib
            # This is the maximum depth d where there exists a node u at depth d
            # with u.original_count >= k AND u is not on the path of word_to_remove.
            # This value can only be max_d_ge_k or second_max_d_ge_k.
            
            max_not_on_path_contrib = 0
            
            if max_d_ge_k > 0:
                 # Check if the single node at max_d_ge_k with count >= k (if unique) is on the path of words[i]
                 is_max_node_unique = (count_nodes_at_d_ge_k.get(max_d_ge_k, 0) == 1)
                 single_node_max_d = single_node_at_d_ge_k.get(max_d_ge_k) # This is the unique node if count was 1

                 is_single_max_node_on_path = False
                 # Check if the single node exists, its depth matches, and it is on the path
                 if is_max_node_unique and single_node_max_d is not None and max_d_ge_k <= L_i:
                      # Get the node on the path of the current word at max_d_ge_k depth
                      # Traverse from root up to max_d_ge_k depth
                      node_at_max_d_on_path = root
                      path_exists_to_max_d = True
                      for dd in range(max_d_ge_k):
                           char_on_path = word_to_remove[dd]
                           if char_on_path not in node_at_max_d_on_path.children:
                                node_at_max_d_on_path = None
                                path_exists_to_max_d = False
                                break
                           node_at_max_d_on_path = node_at_max_d_on_path.children[char_on_path]

                      if path_exists_to_max_d and node_at_max_d_on_path is single_node_max_d:
                           is_single_max_node_on_path = True

                 if is_max_node_unique and is_single_max_node_on_path:
                      # The unique node at max_d_ge_k with count >= k IS on the path of words[i]
                      # So, this node is removed from consideration for "not on path".
                      # The max depth from "not on path" nodes is the second highest.
                      max_not_on_path_contrib = second_max_d_ge_k
                 else:
                      # Case 1: There are multiple nodes at max_d_ge_k with count >= k. At least one is not on path.
                      # Case 2: There is a unique node at max_d_ge_k with count >= k, but it is NOT on the path.
                      # Case 3: There is no unique node at max_d_ge_k with count >= k. (count is 0 or > 1)
                      # In all these cases, the max depth from "not on path" nodes is max_d_ge_k.
                      max_not_on_path_contrib = max_d_ge_k
            
            ans_W = max(max_on_path_contrib, max_not_on_path_contrib)
            answer[i] = ans_W
            answer_by_word[word_to_remove] = ans_W

        return answer