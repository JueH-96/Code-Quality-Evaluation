class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        def common_suffix_length(s1, s2):
            i, j = len(s1) - 1, len(s2) - 1
            length = 0
            while i >= 0 and j >= 0 and s1[i] == s2[j]:
                length += 1
                i -= 1
                j -= 1
            return length
        
        result = []
        
        for query in wordsQuery:
            best_index = 0
            best_suffix_length = common_suffix_length(query, wordsContainer[0])
            best_word_length = len(wordsContainer[0])
            
            for i in range(1, len(wordsContainer)):
                container_word = wordsContainer[i]
                suffix_length = common_suffix_length(query, container_word)
                
                if (suffix_length > best_suffix_length or 
                    (suffix_length == best_suffix_length and len(container_word) < best_word_length)):
                    best_index = i
                    best_suffix_length = suffix_length
                    best_word_length = len(container_word)
            
            result.append(best_index)
        
        return result