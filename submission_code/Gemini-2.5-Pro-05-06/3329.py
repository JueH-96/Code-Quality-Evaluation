from typing import List # Required for List type hint

# Helper classes should be defined outside the Solution class
# as per typical LeetCode contest structure.

class TrieNode:
    def __init__(self):
        # Using dict for children: maps character to TrieNode
        self.children = {} 

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        # No need to mark end of word (e.g., is_end_of_word = True) for this problem.
        # Any path from the root represents a prefix of some inserted word.
    
    def query_longest_match_length(self, word: str) -> int:
        node = self.root
        match_length = 0
        for i, char in enumerate(word):
            if char not in node.children:
                # Current char does not extend any prefix path in the Trie.
                # The loop breaks, and the match_length found so far is returned.
                break 
            
            # Move to the child node corresponding to the current character.
            node = node.children[char]
            
            # If we are here, it means the prefix word[0...i] (length i+1)
            # exists as a path in the Trie. This path corresponds to a prefix 
            # of some word used to build the Trie. Since word[0...i] is also
            # a prefix of the query `word`, it's a common prefix.
            # We update match_length to this new, possibly longer, common prefix length.
            match_length = i + 1
            
        return match_length

class Solution:
  def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
    # Convert integers to their string representations.
    # Using sets for these strings handles duplicate numbers efficiently,
    # as duplicates map to the same string and don't alter the set of unique prefixes.
    # Constraints state 1 <= arr1.length, arr2.length, so arrays are non-empty.
    
    set_str1 = set()
    for x_val in arr1:
        set_str1.add(str(x_val))
    
    set_str2 = set()
    for y_val in arr2:
        set_str2.add(str(y_val))

    # Optimization: build the Trie from the smaller set of unique strings.
    # This minimizes the memory footprint of the Trie.
    # The overall time complexity remains proportional to the sum of unique string lengths.
    if len(set_str1) > len(set_str2):
        # Ensure set_str_build is the smaller (or equally sized) set.
        set_str_build = set_str2  # This set will be used to build the Trie.
        set_str_query = set_str1  # This set will be used to query the Trie.
    else:
        set_str_build = set_str1
        set_str_query = set_str2
    
    # Since arr1 and arr2 are non-empty (per constraints) and contain positive integers,
    # set_str_build and set_str_query will also be non-empty.
    # For example, if arr1=[1], set_str1={"1"}.

    trie = Trie()
    # Insert all strings from the designated build set into the Trie.
    for s_build in set_str_build:
        trie.insert(s_build)

    max_overall_lcp_length = 0
    
    # Iterate through each string in the query set.
    for s_query in set_str_query:
        # For the current query string s_query, find the length of its longest prefix
        # that is also a prefix of some string in set_str_build (those in the Trie).
        current_match_len = trie.query_longest_match_length(s_query)
        
        # Update the overall maximum common prefix length found so far.
        if current_match_len > max_overall_lcp_length:
            max_overall_lcp_length = current_match_len
            
    return max_overall_lcp_length