class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        ans = []
        for query in wordsQuery:
            best_index = -1
            longest_common_suffix_len = 0
            min_len = float('inf')
            
            for i, container_word in enumerate(wordsContainer):
                
                common_suffix_len = 0
                min_length = min(len(query), len(container_word))
                
                for j in range(1, min_length + 1):
                    if query[-j] == container_word[-j]:
                        common_suffix_len += 1
                    else:
                        break
                
                if common_suffix_len > longest_common_suffix_len:
                    longest_common_suffix_len = common_suffix_len
                    best_index = i
                    min_len = len(container_word)
                elif common_suffix_len == longest_common_suffix_len and common_suffix_len > 0:
                    if len(container_word) < min_len:
                        min_len = len(container_word)
                        best_index = i
            
            if best_index == -1:
                longest_common_suffix_len = 0
                min_len = float('inf')
                for i, container_word in enumerate(wordsContainer):
                    if len(container_word) < min_len:
                        min_len = len(container_word)
                        best_index = i
            ans.append(best_index)
        return ans