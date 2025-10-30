import math
from typing import List

# Define TrieNode class
class TrieNode:
    """Represents a node in the Trie."""
    def __init__(self):
        # Children nodes mapping character to TrieNode instance
        self.children = {}
        # Stores a tuple (length, index) representing the best word found so far
        # that passes through this node (i.e., has the suffix corresponding to this node's path).
        # 'Best' is defined by minimum length first, then minimum index as tie-breaker.
        # Initialized to (infinity, infinity) to ensure any valid word info is considered better.
        self.best_info = (math.inf, math.inf) 

class Solution:
    """
    Solves the problem using a Trie built on the reversed strings from wordsContainer.
    This allows efficient searching for longest common suffixes.
    """
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        """
        For each query string in wordsQuery, finds the index of a string in wordsContainer 
        that satisfies the specified criteria based on longest common suffix and tie-breaking rules.

        Args:
            wordsContainer: A list of strings to search within.
            wordsQuery: A list of query strings.

        Returns:
            A list of integers, where each integer is the index of the best matching string 
            in wordsContainer for the corresponding query string in wordsQuery.
        """
        
        # Create the root of the Trie
        root = TrieNode()
        
        # Find the overall best index among all words in wordsContainer based on the criteria:
        # 1. Minimum length
        # 2. Minimum index (for ties in length)
        # This serves as the default result for queries that match no suffixes (or empty suffix).
        min_len_overall = math.inf
        # Initialize min_idx_overall. Since wordsContainer is guaranteed non-empty,
        # this will be updated to a valid index. Start with 0.
        min_idx_overall = 0 

        # Iterate through wordsContainer to find the word meeting the criteria for overall best.
        for j, word in enumerate(wordsContainer):
            length = len(word)
            if length < min_len_overall:
                min_len_overall = length
                min_idx_overall = j
            elif length == min_len_overall:
                # If lengths are equal, choose the word with the smaller index
                if j < min_idx_overall:
                    min_idx_overall = j
        
        # Initialize the root node's best_info with the overall best found.
        # The root represents the empty suffix, shared by all words. Its best_info reflects
        # the best choice considering all words.
        root.best_info = (min_len_overall, min_idx_overall)

        # Build the Trie using reversed words from wordsContainer.
        # During insertion, update the best_info at each node along the path.
        for j, word in enumerate(wordsContainer):
            curr = root
            length = len(word)
            
            # Update root's best_info based on the current word if it's better.
            # This ensures the root always reflects the overall best choice considering all words processed so far.
            # Note: this check is somewhat redundant because we already pre-calculated the overall best and initialized root.
            # However, performing it ensures consistency if the pre-calculation logic was removed.
            if length < curr.best_info[0]:
                curr.best_info = (length, j)
            elif length == curr.best_info[0]:
                 if j < curr.best_info[1]:
                    curr.best_info = (length, j)

            # Traverse the Trie path corresponding to the reversed word.
            # Create nodes if they don't exist. Update node info along the way.
            for char in reversed(word):
                if char not in curr.children:
                    curr.children[char] = TrieNode()
                    # Newly created node starts with default best_info (inf, inf).
                    # The update logic below will set its initial best_info based on the current word `j`.
                
                # Move to the next node in the path
                curr = curr.children[char]
                
                # Update the current node's best_info if word `j` offers a better choice
                # based on the criteria (shorter length, or same length with smaller index).
                if length < curr.best_info[0]:
                    curr.best_info = (length, j)
                elif length == curr.best_info[0]:
                    if j < curr.best_info[1]:
                         curr.best_info = (length, j)

        # Process each query word from wordsQuery.
        ans = []
        for query in wordsQuery:
            curr = root
            # Initialize the answer for this query with the overall best index stored at the root.
            # This handles cases where the query has empty common suffix with all container words,
            # or its reversed path does not extend beyond the root.
            best_match_idx = root.best_info[1] 
            
            # Traverse the Trie using the reversed query string.
            for char in reversed(query):
                # Check if a path exists for the current character.
                if char in curr.children:
                    # Move to the child node.
                    curr = curr.children[char]
                    # This node represents a longer common suffix found. Update the potential answer
                    # with the best index stored at this node, according to the problem criteria.
                    best_match_idx = curr.best_info[1] 
                else:
                    # If no path exists for the character, the longest common suffix path ends here.
                    # The `best_match_idx` already holds the index from the last successfully visited node.
                    break # Stop traversal for this query.
            
            # Append the determined best index for this query to the results list.
            ans.append(best_match_idx)
            
        # Return the list containing the best index for each query.
        return ans