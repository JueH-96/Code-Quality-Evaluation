from collections import defaultdict

class Solution:
    def stringIndices(self, wordsContainer: list[str], wordsQuery: list[str]) -> list[int]:
        # Preprocess the words to build the suffix dictionary
        suffix_dict = defaultdict(dict)
        for idx, word in enumerate(wordsContainer):
            word_len = len(word)
            for l in range(1, word_len + 1):
                s = word[-l:]
                if l not in suffix_dict:
                    suffix_dict[l] = {}
                if s not in suffix_dict[l]:
                    suffix_dict[l][s] = (word_len, idx)
                else:
                    current_len, current_idx = suffix_dict[l][s]
                    if word_len < current_len or (word_len == current_len and idx < current_idx):
                        suffix_dict[l][s] = (word_len, idx)
        
        ans = []
        for s in wordsQuery:
            best_len = float('inf')
            best_index = -1
            max_l = len(s)
            # Check suffixes from longest to shortest
            for l in range(max_l, 0, -1):
                s_suffix = s[-l:]
                if l in suffix_dict and s_suffix in suffix_dict[l]:
                    current_len, current_index = suffix_dict[l][s_suffix]
                    if current_len < best_len or (current_len == best_len and current_index < best_index):
                        best_len = current_len
                        best_index = current_index
            # Check the empty suffix case
            if 0 in suffix_dict and '' in suffix_dict[0]:
                current_len, current_index = suffix_dict[0]['']
                if current_len < best_len or (current_len == best_len and current_index < best_index):
                    best_len = current_len
                    best_index = current_index
            ans.append(best_index)
        return ans