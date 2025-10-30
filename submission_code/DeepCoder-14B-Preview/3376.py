from typing import List

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Preprocess the suffix information
        suffix_info = {}
        for i, word in enumerate(wordsContainer):
            for l in range(1, len(word) + 1):
                s = word[-l:]
                current_length = len(word)
                if s not in suffix_info:
                    suffix_info[s] = (current_length, i)
                else:
                    existing_length, existing_index = suffix_info[s]
                    if current_length < existing_length:
                        suffix_info[s] = (current_length, i)
                    elif current_length == existing_length:
                        if i < existing_index:
                            suffix_info[s] = (current_length, i)
        
        # Compute the minimal length and earliest index for the empty string case
        min_length = float('inf')
        earliest_index = -1
        for i, word in enumerate(wordsContainer):
            word_len = len(word)
            if word_len < min_length:
                min_length = word_len
                earliest_index = i
            elif word_len == min_length:
                if i < earliest_index:
                    earliest_index = i
        
        ans = []
        for query in wordsQuery:
            best_index = -1
            # Generate all possible suffixes of the query, starting from the longest
            for l in range(len(query), 0, -1):
                s = query[-l:]
                if s in suffix_info:
                    best_index = suffix_info[s][1]
                    break
            # If no suffix found, use the minimal length and earliest index
            if best_index == -1:
                best_index = earliest_index
            ans.append(best_index)
        
        return ans