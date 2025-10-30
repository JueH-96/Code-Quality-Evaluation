class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        def common_suffix_length(s1, s2):
            # Find the length of common suffix between two strings
            i = len(s1) - 1
            j = len(s2) - 1
            count = 0
            
            while i >= 0 and j >= 0 and s1[i] == s2[j]:
                count += 1
                i -= 1
                j -= 1
            
            return count
        
        result = []
        
        for query in wordsQuery:
            best_index = 0
            max_suffix_len = 0
            min_word_len = len(wordsContainer[0])
            
            for i, container_word in enumerate(wordsContainer):
                suffix_len = common_suffix_length(container_word, query)
                
                # Update if we found a longer suffix
                if suffix_len > max_suffix_len:
                    max_suffix_len = suffix_len
                    best_index = i
                    min_word_len = len(container_word)
                # If same suffix length, check if this word is shorter
                elif suffix_len == max_suffix_len:
                    if len(container_word) < min_word_len:
                        best_index = i
                        min_word_len = len(container_word)
            
            result.append(best_index)
        
        return result