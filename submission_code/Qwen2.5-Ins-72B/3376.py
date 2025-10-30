from typing import List

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Preprocess wordsContainer to store suffixes and their indices
        suffix_map = {}
        for idx, word in enumerate(wordsContainer):
            for i in range(len(word) + 1):
                suffix = word[i:]
                if suffix not in suffix_map:
                    suffix_map[suffix] = (idx, len(word))
                else:
                    # Update if the current word is shorter or has the same length but comes earlier
                    if len(word) < suffix_map[suffix][1] or (len(word) == suffix_map[suffix][1] and idx < suffix_map[suffix][0]):
                        suffix_map[suffix] = (idx, len(word))
        
        # Function to find the longest common suffix
        def longest_common_suffix(word1, word2):
            min_len = min(len(word1), len(word2))
            for i in range(min_len):
                if word1[-(i + 1)] != word2[-(i + 1)]:
                    return word1[-i:] if i > 0 else ""
            return word1[-min_len:]
        
        # Process each query
        result = []
        for query in wordsQuery:
            best_idx = -1
            best_len = float('inf')
            for suffix in suffix_map:
                if query.endswith(suffix):
                    idx, length = suffix_map[suffix]
                    if length < best_len or (length == best_len and idx < best_idx):
                        best_idx = idx
                        best_len = length
            if best_idx == -1:
                # If no common suffix, find the shortest word in wordsContainer
                for idx, word in enumerate(wordsContainer):
                    if len(word) < best_len or (len(word) == best_len and idx < best_idx):
                        best_idx = idx
                        best_len = len(word)
            result.append(best_idx)
        
        return result