class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Preprocess wordsContainer to store their lengths and indices
        container_info = [(word, len(word), idx) for idx, word in enumerate(wordsContainer)]
        
        ans = []
        for query in wordsQuery:
            max_suffix_len = -1
            candidates = []
            for word, length, idx in container_info:
                # Find the length of the longest common suffix
                min_len = min(len(query), len(word))
                suffix_len = 0
                for i in range(1, min_len + 1):
                    if query[-i] == word[-i]:
                        suffix_len += 1
                    else:
                        break
                if suffix_len > max_suffix_len:
                    max_suffix_len = suffix_len
                    candidates = [(length, idx)]
                elif suffix_len == max_suffix_len:
                    candidates.append((length, idx))
            # Now, select the candidate with the smallest length, and earliest index if lengths are same
            if not candidates:
                # If no common suffix, choose the one with the smallest length and earliest index
                candidates = [(length, idx) for word, length, idx in container_info]
                min_len = min(candidates, key=lambda x: x[0])[0]
                candidates = [c for c in candidates if c[0] == min_len]
                # Choose the earliest index
                selected = min(candidates, key=lambda x: x[1])[1]
            else:
                min_len = min(candidates, key=lambda x: x[0])[0]
                candidates = [c for c in candidates if c[0] == min_len]
                selected = min(candidates, key=lambda x: x[1])[1]
            ans.append(selected)
        return ans