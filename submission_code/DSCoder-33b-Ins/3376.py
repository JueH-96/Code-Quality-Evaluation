class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        def common_suffix(s1, s2):
            i, j = len(s1) - 1, len(s2) - 1
            while i >= 0 and j >= 0:
                if s1[i] != s2[j]:
                    break
                i -= 1
                j -= 1
            return s1[i+1:]

        ans = []
        for word in wordsQuery:
            max_suffix = ''
            max_index = -1
            for i, container_word in enumerate(wordsContainer):
                suffix = common_suffix(word, container_word)
                if len(suffix) > len(max_suffix) or (len(suffix) == len(max_suffix) and container_word < wordsContainer[max_index]):
                    max_suffix = suffix
                    max_index = i
            ans.append(max_index)
        return ans