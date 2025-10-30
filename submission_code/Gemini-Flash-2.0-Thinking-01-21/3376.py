class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        ans = []
        for query_word in wordsQuery:
            best_index = -1
            max_suffix_length = -1
            min_length = float('inf')
            for i in range(len(wordsContainer)):
                container_word = wordsContainer[i]
                suffix_length = 0
                min_len = min(len(query_word), len(container_word))
                for j in range(1, min_len + 1):
                    if query_word[-j:] == container_word[-j:]:
                        suffix_length = j
                    else:
                        break
                if suffix_length > max_suffix_length:
                    max_suffix_length = suffix_length
                    best_index = i
                    min_length = len(container_word)
                elif suffix_length == max_suffix_length:
                    if best_index == -1:
                        best_index = i
                        min_length = len(container_word)
                    elif len(container_word) < min_length:
                        best_index = i
                        min_length = len(container_word)
                    elif len(container_word) == min_length:
                        if best_index == -1 or i < best_index:
                            best_index = i
                            min_length = len(container_word)
                            
            if best_index == -1: # This case should not happen based on problem description, but for robustness
                best_index = 0 # If no common suffix, choose the first one as per example 1 explanation for "xyz"

            current_max_suffix_len = -1
            current_min_len = float('inf')
            current_best_index = -1
            
            for i in range(len(wordsContainer)):
                container_word = wordsContainer[i]
                suffix_length = 0
                min_len = min(len(query_word), len(container_word))
                for j in range(1, min_len + 1):
                    if query_word[-j:] == container_word[-j:]:
                        suffix_length = j
                    else:
                        break
                if suffix_length > current_max_suffix_len:
                    current_max_suffix_len = suffix_length
                    current_best_index = i
                    current_min_len = len(container_word)
                elif suffix_length == current_max_suffix_len:
                    if current_best_index == -1:
                        current_best_index = i
                        current_min_len = len(container_word)
                    elif len(container_word) < current_min_len:
                        current_best_index = i
                        current_min_len = len(container_word)
                    elif len(container_word) == current_min_len:
                        if current_best_index == -1 or i < current_best_index:
                            current_best_index = i
                            current_min_len = len(container_word)

            best_index = -1
            max_suffix_length = -1
            min_len_for_max_suffix = float('inf')
            
            candidates_indices = []
            
            for i in range(len(wordsContainer)):
                container_word = wordsContainer[i]
                suffix_length = 0
                min_len_calc = min(len(query_word), len(container_word))
                for j in range(1, min_len_calc + 1):
                    if query_word[-j:] == container_word[-j:]:
                        suffix_length = j
                    else:
                        break
                if suffix_length > max_suffix_length:
                    max_suffix_length = suffix_length
                    candidates_indices = [i]
                    min_len_for_max_suffix = len(container_word)
                elif suffix_length == max_suffix_length:
                    if max_suffix_length > -1: # avoid adding when max_suffix_length is still -1 and suffix_length is 0
                        candidates_indices.append(i)
                        if len(container_word) < min_len_for_max_suffix:
                            min_len_for_max_suffix = len(container_word)


            final_best_index = -1
            final_min_len = float('inf')

            if not candidates_indices: # Handle case when no common suffix at all
                for i in range(len(wordsContainer)):
                    if len(wordsContainer[i]) < final_min_len:
                        final_min_len = len(wordsContainer[i])
                        final_best_index = i
                    elif len(wordsContainer[i]) == final_min_len:
                        if final_best_index == -1 or i < final_best_index:
                            final_best_index = i
                ans.append(final_best_index)
            else:
                valid_indices = []
                for index in candidates_indices:
                    if len(wordsContainer[index]) == min_len_for_max_suffix:
                        valid_indices.append(index)
                if valid_indices:
                    ans.append(min(valid_indices))
                else: # If no word with min length for max suffix, then find min length among all with max suffix
                    min_len_in_candidates = float('inf')
                    best_candidate_index = -1
                    for index in candidates_indices:
                        if len(wordsContainer[index]) < min_len_in_candidates:
                            min_len_in_candidates = len(wordsContainer[index])
                            best_candidate_index = index
                        elif len(wordsContainer[index]) == min_len_in_candidates:
                            if best_candidate_index == -1 or index < best_candidate_index:
                                best_candidate_index = index
                    ans.append(best_candidate_index)


        return ans