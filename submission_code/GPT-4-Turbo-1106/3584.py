class Solution:
    def validSequence(self, word1: str, word2: str) -> list[int]:
        n, m = len(word1), len(word2)
        
        # Helper function to check if word2 can be formed from word1 using indices in seq
        def is_almost_equal(seq):
            changes = 0
            for i in range(m):
                if word1[seq[i]] != word2[i]:
                    changes += 1
                    if changes > 1:
                        return False
            return True
        
        # Helper function to find the lexicographically smallest sequence
        def find_sequence(start, length, seq):
            if length == m:
                if is_almost_equal(seq):
                    return seq
                return None
            for i in range(start, n - (m - length) + 1):
                next_seq = find_sequence(i + 1, length + 1, seq + [i])
                if next_seq is not None:
                    return next_seq
            return None
        
        # Start the search from the beginning of word1
        return find_sequence(0, 0, []) or []