class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        ans = []
        
        for query in wordsQuery:
            max_suffix_length = 0
            best_index = -1
            
            for index, word in enumerate(wordsContainer):
                # Find the length of the common suffix
                suffix_length = 0
                while (suffix_length < len(word) and 
                       suffix_length < len(query) and 
                       word[-(suffix_length + 1)] == query[-(suffix_length + 1)]):
                    suffix_length += 1
                
                # Check if we found a longer suffix or if we need to update the best index
                if suffix_length > max_suffix_length:
                    max_suffix_length = suffix_length
                    best_index = index
                elif suffix_length == max_suffix_length:
                    # If the same length, prefer the shorter word
                    if best_index == -1 or (len(word) < len(wordsContainer[best_index])):
                        best_index = index
            
            ans.append(best_index)
        
        return ans