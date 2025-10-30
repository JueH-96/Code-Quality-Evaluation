class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n = len(word2)
        m = len(word1)
        ans = []
        diff = 0
        indices = []
        
        for i in range(m):
            if diff < n and word1[i] == word2[diff]:
                indices.append(i)
                diff += 1

        if diff != n:
            return []

        
        res = []
        diff_count = 0
        
        for i in range(n):
            if word1[indices[i]] != word2[i]:
                diff_count +=1

        if diff_count > 1:
            return []

        
        
        import itertools
        
        min_seq = []
        min_diff = float('inf')

        for i in range(1 << len(indices)):
            current_indices = []
            temp_word = ""
            for j in range(len(indices)):
                if (i >> j) & 1:
                    current_indices.append(indices[j])
                    temp_word += word1[indices[j]]

            if len(current_indices) != n:
                continue

            diff_count = 0
            for k in range(n):
                if temp_word[k] != word2[k]:
                    diff_count += 1

            if diff_count <=1:
                if len(current_indices) < min_diff:
                    min_diff = len(current_indices)
                    min_seq = current_indices
                elif len(current_indices) == min_diff:
                    if current_indices < min_seq:
                        min_seq = current_indices

        return min_seq