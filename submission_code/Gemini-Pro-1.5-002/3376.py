class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        ans = []
        for query in wordsQuery:
            max_suffix_len = 0
            best_index = -1
            for i, container_word in enumerate(wordsContainer):
                current_suffix_len = 0
                for j in range(1, min(len(query), len(container_word)) + 1):
                    if query[-j:] == container_word[-j:]:
                        current_suffix_len = j
                    else:
                        break
                
                if current_suffix_len > max_suffix_len:
                    max_suffix_len = current_suffix_len
                    best_index = i
                elif current_suffix_len == max_suffix_len and current_suffix_len > 0:
                    if len(wordsContainer[best_index]) > len(container_word):
                        best_index = i
            
            if best_index == -1:
                min_len = float('inf')
                for i, container_word in enumerate(wordsContainer):
                    if len(container_word) < min_len:
                        min_len = len(container_word)
                        best_index = i

            ans.append(best_index)
        return ans