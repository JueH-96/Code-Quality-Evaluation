class Solution:
    def validSequence(self, word1: str, word2: str) -> list[int]:
        n1 = len(word1)
        n2 = len(word2)

        # Try for 0 changes
        indices_0_change = []
        last_index_0 = -1
        possible_0_change = True
        for char2 in word2:
            found_index = False
            for j in range(last_index_0 + 1, n1):
                if word1[j] == char2:
                    indices_0_change.append(j)
                    last_index_0 = j
                    found_index = True
                    break
            if not found_index:
                possible_0_change = False
                break
        if possible_0_change:
            return indices_0_change

        # Try for 1 change
        for change_pos in range(n2):
            indices_1_change = []
            last_index_1 = -1
            change_used = False
            possible_1_change = True
            for i in range(n2):
                char2 = word2[i]
                found_index = False
                for j in range(last_index_1 + 1, n1):
                    if i == change_pos:
                        if word1[j] != char2:
                            indices_1_change.append(j)
                            last_index_1 = j
                            found_index = True
                            change_used = True
                            break
                    else:
                        if word1[j] == char2:
                            indices_1_change.append(j)
                            last_index_1 = j
                            found_index = True
                            break
                if not found_index:
                    possible_1_change = False
                    break
            if possible_1_change and change_used:
                return indices_1_change

        return []