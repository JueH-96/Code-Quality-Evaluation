class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        def common_suffix_length(s1, s2):
            n1 = len(s1)
            n2 = len(s2)
            length = 0
            for i in range(1, min(n1, n2) + 1):
                if s1[n1 - i:] == s2[n2 - i:]:
                    length = i
                else:
                    break
            return length

        ans = []
        for query_word in wordsQuery:
            max_suffix_length = 0
            best_index = -1
            for i in range(len(wordsContainer)):
                container_word = wordsContainer[i]
                current_suffix_length = common_suffix_length(container_word, query_word)
                if current_suffix_length > max_suffix_length:
                    max_suffix_length = current_suffix_length
                    best_index = i
                elif current_suffix_length == max_suffix_length:
                    if best_index == -1:
                        best_index = i
                    else:
                        len_current = len(container_word)
                        len_best = len(wordsContainer[best_index])
                        if len_current < len_best:
                            best_index = i
                        elif len_current == len_best and i < best_index:
                            best_index = i
            ans.append(best_index)
        return ans