class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        def longest_common_suffix_length(s1, s2):
            # Find the longest common suffix length between two strings
            min_len = min(len(s1), len(s2))
            lcs_length = 0
            for i in range(1, min_len + 1):
                if s1[-i] == s2[-i]:
                    lcs_length += 1
                else:
                    break
            return lcs_length
        
        result = []
        
        for query in wordsQuery:
            max_suffix_length = -1
            best_index = -1
            best_length = float('inf')
            
            for index, word in enumerate(wordsContainer):
                lcs_length = longest_common_suffix_length(word, query)
                
                if lcs_length > max_suffix_length:
                    max_suffix_length = lcs_length
                    best_index = index
                    best_length = len(word)
                elif lcs_length == max_suffix_length:
                    if len(word) < best_length:
                        best_index = index
                        best_length = len(word)
            
            result.append(best_index)
        
        return result