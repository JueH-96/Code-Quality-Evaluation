class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        m, n = len(word2), len(word1)
        
        # Function to find exact match subsequence
        def find_exact():
            seq = []
            j = 0
            for i in range(n):
                if word1[i] == word2[j]:
                    seq.append(i)
                    j += 1
                    if j == m:
                        return seq
            return None if j == m else []
        
        # Find exact match
        exact_seq = find_exact()
        if exact_seq:
            return exact_seq
        
        # Function to find subsequence with one mismatch at position 'mismatch_pos' in word2
        def find_with_one_mismatch(mismatch_pos):
            seq = []
            j = 0
            used_mismatch = False
            for i in range(n):
                if j == m:
                    break
                if word1[i] == word2[j]:
                    seq.append(i)
                    j += 1
                else:
                    if not used_mismatch and j == mismatch_pos:
                        seq.append(i)
                        used_mismatch = True
                        j += 1
                    else:
                        continue
            return seq if j == m else None
        
        # Try all possible positions for the mismatch and find the lex smallest sequence
        min_seq = None
        for mismatch_pos in range(m):
            seq = find_with_one_mismatch(mismatch_pos)
            if seq is not None:
                if min_seq is None or seq < min_seq:
                    min_seq = seq
        return min_seq if min_seq else []