class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Build trie for suffixes
        trie = {}
        for i, word in enumerate(wordsContainer):
            curr = trie
            # Store index and length for each node
            if '#' not in curr:
                curr['#'] = (i, len(word))
            else:
                if len(word) < curr['#'][1]:
                    curr['#'] = (i, len(word))
                    
            # Insert word in reverse
            for c in word[::-1]:
                if c not in curr:
                    curr[c] = {}
                curr = curr[c]
                # Store index and length for each node
                if '#' not in curr:
                    curr['#'] = (i, len(word))
                else:
                    if len(word) < curr['#'][1]:
                        curr['#'] = (i, len(word))
        
        ans = []
        # Process each query
        for query in wordsQuery:
            curr = trie
            # Try to match longest suffix
            for c in query[::-1]:
                if c not in curr:
                    break
                curr = curr[c]
            # Return index stored at last matched node
            ans.append(curr['#'][0])
            
        return ans