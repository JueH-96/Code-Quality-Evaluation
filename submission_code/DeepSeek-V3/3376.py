class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Preprocess wordsContainer to store indices based on suffix lengths
        # We will use a dictionary where the key is the suffix and the value is a list of tuples (length, index)
        suffix_map = {}
        
        for idx, word in enumerate(wordsContainer):
            n = len(word)
            for l in range(n + 1):
                suffix = word[-l:] if l > 0 else ""
                if suffix not in suffix_map:
                    suffix_map[suffix] = []
                suffix_map[suffix].append((n, idx))
        
        # For each query, find the longest common suffix
        ans = []
        for query in wordsQuery:
            max_len = 0
            best_candidate = None
            # Iterate over possible suffix lengths
            for l in range(len(query) + 1):
                suffix = query[-l:] if l > 0 else ""
                if suffix in suffix_map:
                    if l > max_len:
                        max_len = l
                        # Among all candidates with this suffix, find the one with the smallest length and earliest index
                        candidates = suffix_map[suffix]
                        min_length = float('inf')
                        best_index = -1
                        for length, index in candidates:
                            if length < min_length:
                                min_length = length
                                best_index = index
                            elif length == min_length:
                                if index < best_index:
                                    best_index = index
                        best_candidate = best_index
            if best_candidate is not None:
                ans.append(best_candidate)
            else:
                # If no common suffix, choose the shortest word in wordsContainer
                min_length = float('inf')
                best_index = -1
                for idx, word in enumerate(wordsContainer):
                    if len(word) < min_length:
                        min_length = len(word)
                        best_index = idx
                ans.append(best_index)
        return ans