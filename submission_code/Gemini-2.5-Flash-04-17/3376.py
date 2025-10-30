from typing import List

class TrieNode:
    def __init__(self):
        self.children = {} # char -> TrieNode
        # Stores the index and length of the best container string
        # whose reversed form passes through this node.
        # "Best" means shortest length, then smallest original index.
        # Initialize with a value indicating no string passes through yet.
        self.min_len_index = (-1, float('inf')) # (index, length)

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        root = TrieNode()
        n_container = len(wordsContainer)

        # Build Trie with reversed words and store min_len_index at each node
        for i in range(n_container):
            word = wordsContainer[i]
            word_rev = word[::-1]
            word_len = len(word)

            # Update root's best string info
            # The root node corresponds to the empty suffix.
            # The best string for the empty suffix is the shortest one overall (earliest among shortest).
            # We can use tuple comparison directly: (length, index)
            # We want the minimum (length, index).
            if word_len < root.min_len_index[1] or (word_len == root.min_len_index[1] and i < root.min_len_index[0]):
                 root.min_len_index = (i, word_len)
            # Or using min function with key:
            # root.min_len_index = min(root.min_len_index, (i, word_len), key=lambda x: (x[1], x[0]))


            current_node = root
            for char in word_rev:
                if char not in current_node.children:
                    current_node.children[char] = TrieNode()
                current_node = current_node.children[char]

                # Update current node's best string info.
                # This node represents a specific suffix.
                # We want the shortest string (earliest among shortest) that has this suffix.
                if word_len < current_node.min_len_index[1] or (word_len == current_node.min_len_index[1] and i < current_node.min_len_index[0]):
                    current_node.min_len_index = (i, word_len)
                # Or using min function with key:
                # current_node.min_len_index = min(current_node.min_len_index, (i, word_len), key=lambda x: (x[1], x[0]))


        ans = []
        for query in wordsQuery:
            query_rev = query[::-1]
            current_node = root

            # best_match_info = (lcs_len, string_len, index)
            # Initialize with the best string overall (corresponds to empty suffix lcs_len=0)
            # This is the fallback if no common suffix > 0 is found, or if the best
            # string for the maximal common suffix happens to be worse than the
            # best string overall based on length/index criteria.
            # The problem prioritizes LCS length first. So the initial best_match_info
            # should represent the best string *found so far*, starting with the empty suffix case.
            best_match_info = (0, root.min_len_index[1], root.min_len_index[0])

            # Traverse query_rev in the Trie
            for k, char in enumerate(query_rev):
                if char in current_node.children:
                    current_node = current_node.children[char]
                    # The path length `k+1` corresponds to the length of the common suffix
                    # between the query and any string whose reversed form follows this path.
                    current_lcs_len = k + 1

                    # Get the best candidate string info from the current node.
                    # This string is the shortest (earliest among shortest) that has the
                    # suffix corresponding to the current path of length `current_lcs_len`.
                    candidate_index = current_node.min_len_index[0]
                    candidate_len = current_node.min_len_index[1]

                    # Compare the current candidate match with the best match found so far.
                    # Criteria: Maximize lcs_len, then minimize string_len, then minimize index.
                    # We compare the tuple (current_lcs_len, candidate_len, candidate_index)
                    # with `best_match_info = (best_lcs_len, best_string_len, best_index)`.

                    # If the current LCS length is greater, it's strictly better.
                    if current_lcs_len > best_match_info[0]:
                        best_match_info = (current_lcs_len, candidate_len, candidate_index)
                    # If the current LCS length is equal to the best found so far,
                    # apply the secondary and tertiary criteria (min string_len, min index).
                    elif current_lcs_len == best_match_info[0]:
                        # Compare string lengths
                        if candidate_len < best_match_info[1]:
                            best_match_info = (current_lcs_len, candidate_len, candidate_index)
                        # If string lengths are also equal, compare indices.
                        # The index stored at `current_node.min_len_index[0]` is the earliest
                        # index for the shortest string having this specific suffix path.
                        # `best_match_info[2]` is the earliest index found so far among *all*
                        # candidates that achieved the same maximal LCS length and minimal string length.
                        # So, we update if the current candidate's index is strictly smaller.
                        elif candidate_len == best_match_info[1]:
                             if candidate_index < best_match_info[2]:
                                 best_match_info = (current_lcs_len, candidate_len, candidate_index)


                else:
                    # Cannot extend the common suffix match using the Trie.
                    # The longest possible common suffix found by traversing this path
                    # is the length `k` of the path taken successfully so far.
                    # The `best_match_info` already holds the best result based on
                    # common suffix lengths up to `k`.
                    break # Stop traversing this query

            # After traversing the query reversed string as far as possible,
            # `best_match_info` contains the info about the best match found
            # based on the criteria (max lcs_len, min string_len, min index).
            ans.append(best_match_info[2])

        return ans