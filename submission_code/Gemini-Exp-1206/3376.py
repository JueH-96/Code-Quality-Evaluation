class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        ans = []
        for query in wordsQuery:
            max_suffix_len = -1
            min_word_len = float('inf')
            best_index = -1
            for i, word in enumerate(wordsContainer):
                suffix_len = 0
                for j in range(min(len(query), len(word))):
                    if query[len(query) - 1 - j] == word[len(word) - 1 - j]:
                        suffix_len += 1
                    else:
                        break
                if suffix_len > max_suffix_len:
                    max_suffix_len = suffix_len
                    min_word_len = len(word)
                    best_index = i
                elif suffix_len == max_suffix_len:
                    if len(word) < min_word_len:
                        min_word_len = len(word)
                        best_index = i
                    elif len(word) == min_word_len:
                        if best_index == -1:
                            best_index = i

            if best_index == -1:
                
                min_len = float('inf')
                best_idx = -1
                for idx, w in enumerate(wordsContainer):
                    if len(w) < min_len:
                        min_len = len(w)
                        best_idx = idx
                ans.append(best_idx)
            else:
                ans.append(best_index)
        return ans