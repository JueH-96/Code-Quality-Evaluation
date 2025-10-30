class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n = len(word1)
        m = len(word2)
        result = []

        def is_almost_equal(s1, s2):
            diff = 0
            for i in range(len(s1)):
                if s1[i] != s2[i]:
                    diff += 1
            return diff <= 1

        def find_sequence(index, current_sequence):
            nonlocal result
            if len(current_sequence) == m:
                temp_word = ""
                for i in current_sequence:
                    temp_word += word1[i]
                if is_almost_equal(temp_word, word2):
                    if not result or current_sequence < result:
                        result = current_sequence[:]
                return

            if index >= n:
                return

            find_sequence(index + 1, current_sequence)
            
            current_sequence.append(index)
            find_sequence(index + 1, current_sequence)
            current_sequence.pop()

        find_sequence(0, [])
        return result