class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        def get_common_suffix_length(s1, s2):
            # Find length of common suffix between s1 and s2
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
            best_suffix_len = get_common_suffix_length(query, wordsContainer[0])
            best_word_len = len(wordsContainer[0])
            
            for i in range(1, len(wordsContainer)):
                container_word = wordsContainer[i]
                suffix_len = get_common_suffix_length(query, container_word)
                word_len = len(container_word)
                
                # Update best if:
                # 1. Longer common suffix, OR
                # 2. Same suffix length but shorter word, OR  
                # 3. Same suffix length and same word length (earlier index wins automatically)
                if (suffix_len > best_suffix_len or 
                    (suffix_len == best_suffix_len and word_len < best_word_len)):
                    best_index = i
                    best_suffix_len = suffix_len
                    best_word_len = word_len
            
            result.append(best_index)
        
        return result