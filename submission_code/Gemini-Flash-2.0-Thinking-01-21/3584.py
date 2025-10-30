class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n1 = len(word1)
        n2 = len(word2)
        
        def is_almost_equal(indices):
            sub_word = ""
            for index in indices:
                sub_word += word1[index]
            
            diff_count = 0
            for i in range(n2):
                if sub_word[i] != word2[i]:
                    diff_count += 1
            return diff_count <= 1

        def solve(index_w2, last_index_w1, current_sequence, diff_count):
            if index_w2 == n2:
                if diff_count <= 1:
                    return current_sequence
                else:
                    return None

            for index_w1 in range(last_index_w1 + 1, n1):
                next_diff_count = diff_count
                if word1[index_w1] != word2[index_w2]:
                    next_diff_count += 1
                if next_diff_count <= 1:
                    result = solve(index_w2 + 1, index_w1, current_sequence + [index_w1], next_diff_count)
                    if result:
                        return result
            return None

        result_indices = solve(0, -1, [], 0)
        if result_indices:
            return result_indices
        else:
            return []