class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        ans = []
        for query in wordsQuery:
            best_len = -1
            best_index = -1
            best_str_len = float('inf')
            for i, container_str in enumerate(wordsContainer):
                len_suffix = 0
                min_len = min(len(query), len(container_str))
                for j in range(1, min_len + 1):
                    if query[-j:] == container_str[-j:]:
                        len_suffix = j
                if len_suffix > best_len:
                    best_len = len_suffix
                    best_index = i
                    best_str_len = len(container_str)
                elif len_suffix == best_len:
                    if len(container_str) < best_str_len:
                        best_index = i
                        best_str_len = len(container_str)

            ans.append(best_index)
        return ans