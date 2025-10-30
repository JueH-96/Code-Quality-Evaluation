class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n1, n2 = len(word1), len(word2)
        if n2 >= n1:
            return []
            
        def check_almost_equal(indices):
            # Get string formed by indices from word1
            s = ''
            for i in indices:
                s += word1[i]
                
            # Count differences between s and word2
            diff = 0
            for i in range(n2):
                if s[i] != word2[i]:
                    diff += 1
                if diff > 1:
                    return False
            return True
            
        def backtrack(curr_indices, start, remaining):
            if remaining == 0:
                if check_almost_equal(curr_indices):
                    return curr_indices
                return None
                
            for i in range(start, n1):
                curr_indices.append(i)
                result = backtrack(curr_indices, i + 1, remaining - 1)
                if result is not None:
                    return result
                curr_indices.pop()
                
            return None
            
        # Try to find valid sequence
        result = backtrack([], 0, n2)
        return result if result is not None else []