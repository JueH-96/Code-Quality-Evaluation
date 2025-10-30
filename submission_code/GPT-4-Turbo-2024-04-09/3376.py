class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        def longest_common_suffix_length(s1, s2):
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
            max_suffix_length = -1
            best_index = -1
            best_length = float('inf')
            
            for index, container in enumerate(wordsContainer):
                suffix_length = longest_common_suffix_length(container, query)
                
                if suffix_length > max_suffix_length:
                    max_suffix_length = suffix_length
                    best_index = index
                    best_length = len(container)
                elif suffix_length == max_suffix_length:
                    if len(container) < best_length:
                        best_index = index
                        best_length = len(container)
                    elif len(container) == best_length:
                        if index < best_index:
                            best_index = index
            
            result.append(best_index)
        
        return result