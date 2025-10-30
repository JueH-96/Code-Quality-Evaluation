class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        def longest_common_suffix(s1, s2):
            i = len(s1) - 1
            j = len(s2) - 1
            count = 0
            while i >= 0 and j >= 0 and s1[i] == s2[j]:
                count += 1
                i -= 1
                j -= 1
            return count

        ans = []
        for query in wordsQuery:
            best_match_index = -1
            max_common_suffix = -1
            best_match_length_container = -1

            for i, container_word in enumerate(wordsContainer):
                common_suffix_length = longest_common_suffix(container_word, query)

                if common_suffix_length > max_common_suffix:
                    max_common_suffix = common_suffix_length
                    best_match_index = i
                    best_match_length_container = len(container_word)
                elif common_suffix_length == max_common_suffix and max_common_suffix != -1:
                    if len(container_word) < best_match_length_container:
                        best_match_index = i
                        best_match_length_container = len(container_word)
                    elif len(container_word) == best_match_length_container:
                        pass  # Keep the earlier index
                elif best_match_index == -1:
                    max_common_suffix = common_suffix_length
                    best_match_index = i
                    best_match_length_container = len(container_word)

            # Handle the case where no common suffix was found (max_common_suffix remains -1)
            if max_common_suffix == -1:
                min_len = float('inf')
                shortest_index = -1
                for i, word in enumerate(wordsContainer):
                    if len(word) < min_len:
                        min_len = len(word)
                        shortest_index = i
                ans.append(shortest_index)
            else:
                # Find the best match among those with the longest common suffix
                candidates = []
                for i, container_word in enumerate(wordsContainer):
                    if longest_common_suffix(container_word, query) == max_common_suffix:
                        candidates.append((i, container_word))

                if not candidates:
                    # This should not happen
                    raise Exception("Unexpected state")

                best_idx = candidates[0][0]
                min_len = len(candidates[0][1])

                for idx, word in candidates[1:]:
                    if len(word) < min_len:
                        min_len = len(word)
                        best_idx = idx
                    elif len(word) == min_len:
                        pass # Keep the earlier index

                ans.append(best_idx)

        return ans