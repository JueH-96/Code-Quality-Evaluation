class TrieNode:
    def __init__(self):
        self.children = {}
        self.best_index = None
        self.best_length = float('inf')

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Reverse the words in wordsContainer and wordsQuery
        reversedWordsContainer = [word[::-1] for word in wordsContainer]
        reversedWordsQuery = [word[::-1] for word in wordsQuery]
        
        root = TrieNode()
        n = len(wordsContainer)
        
        # Build the trie
        for idx, word in enumerate(reversedWordsContainer):
            curr = root
            word_length = len(wordsContainer[idx])
            # Update root node's best index and length
            if word_length < root.best_length or (word_length == root.best_length and idx < root.best_index):
                root.best_index = idx
                root.best_length = word_length
            curr = root
            for char in word:
                if char not in curr.children:
                    curr.children[char] = TrieNode()
                curr = curr.children[char]
                # Update the best index and length at this node
                if word_length < curr.best_length or (word_length == curr.best_length and idx < curr.best_index):
                    curr.best_index = idx
                    curr.best_length = word_length
        
        # Process each query
        ans = []
        for query in reversedWordsQuery:
            curr = root
            max_match_length = 0
            best_index = root.best_index
            match_length = 0
            for char in query:
                if char in curr.children:
                    curr = curr.children[char]
                    match_length += 1
                    # Update best index if we have a longer match (which always increases)
                    max_match_length = match_length
                    best_index = curr.best_index
                else:
                    break
            ans.append(best_index)
        
        return ans