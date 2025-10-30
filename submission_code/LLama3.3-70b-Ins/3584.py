from typing import List

class Solution:
    def findLUSlength(self, word1: str, word2: str) -> int:
        if word1 == word2:
            return -1
        else:
            return max(len(word1), len(word2))

    def validSequence(self, word1: str, word2: str) -> List[int]:
        def is_almost_equal(s1: str, s2: str) -> bool:
            diff_count = 0
            for c1, c2 in zip(s1, s2):
                if c1 != c2:
                    diff_count += 1
                if diff_count > 1:
                    return False
            return True

        def get_sequence(word1: str, word2: str) -> List[int]:
            sequence = []
            i = 0
            for c in word2:
                while i < len(word1) and word1[i] != c:
                    i += 1
                if i == len(word1):
                    return []
                sequence.append(i)
                i += 1
            return sequence

        min_sequence = []
        for i in range(len(word1)):
            for j in range(i + 1, len(word1) + 1):
                sub_word1 = ''.join([word1[k] for k in range(i, j)])
                if len(sub_word1) == len(word2) and is_almost_equal(sub_word1, word2):
                    sequence = get_sequence(word1, word2)
                    if sequence and (not min_sequence or sequence < min_sequence):
                        min_sequence = sequence
        return min_sequence