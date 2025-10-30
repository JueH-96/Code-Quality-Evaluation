class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n = len(word1)
        m = len(word2)
        best_sequence = None

        def is_almost_equal(s1, s2):
            if len(s1) != len(s2):
                return False
            diff = 0
            for char1, char2 in zip(s1, s2):
                if char1 != char2:
                    diff += 1
            return diff <= 1

        def find_valid_sequences(index, start, current_sequence):
            nonlocal best_sequence

            if len(current_sequence) == m:
                sub_word = "".join(word1[i] for i in current_sequence)
                if is_almost_equal(sub_word, word2):
                    if best_sequence is None or current_sequence < best_sequence:
                        best_sequence = list(current_sequence)
                return

            if index == m:
                return

            for i in range(start, n):
                current_sequence.append(i)
                find_valid_sequences(index + 1, i + 1, current_sequence)
                current_sequence.pop()

        find_valid_sequences(0, 0, [])
        return best_sequence if best_sequence is not None else []