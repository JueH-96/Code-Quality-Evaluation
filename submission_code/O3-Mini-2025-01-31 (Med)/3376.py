from typing import List

class TrieNode:
    def __init__(self):
        self.children = {}
        # candidate is a tuple: (word_length, index)
        self.candidate = None

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Build the Trie from the reversed wordsContainer.
        root = TrieNode()
        
        # Precompute the best candidate over all words (for common suffix length = 0)
        best0 = None
        for idx, word in enumerate(wordsContainer):
            candidate = (len(word), idx)
            if best0 is None or candidate < best0:
                best0 = candidate
        # set the candidate for root to best overall candidate.
        root.candidate = best0
        
        # Insert each container word into trie in reversed order.
        for idx, word in enumerate(wordsContainer):
            current = root
            candidate = (len(word), idx)
            # update candidate for the current node if needed
            if current.candidate is None or candidate < current.candidate:
                current.candidate = candidate
            for ch in reversed(word):
                if ch not in current.children:
                    current.children[ch] = TrieNode()
                current = current.children[ch]
                # update candidate for this node
                if current.candidate is None or candidate < current.candidate:
                    current.candidate = candidate
        
        # Process each query word.
        result = []
        for q in wordsQuery:
            current = root
            best_candidate = current.candidate  # start with best candidate at root (common suffix = "")
            # Traverse through query reversed.
            for ch in reversed(q):
                if ch in current.children:
                    current = current.children[ch]
                    best_candidate = current.candidate
                else:
                    break
            # best_candidate is a tuple (length, index)
            result.append(best_candidate[1])
        return result

# Sample test cases
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    wordsContainer = ["abcd", "bcd", "xbcd"]
    wordsQuery = ["cd", "bcd", "xyz"]
    print(sol.stringIndices(wordsContainer, wordsQuery))  # Expected [1,1,1]
    
    # Example 2:
    wordsContainer = ["abcdefgh", "poiuygh", "ghghgh"]
    wordsQuery = ["gh", "acbfgh", "acbfegh"]
    print(sol.stringIndices(wordsContainer, wordsQuery))  # Expected [2,0,2]