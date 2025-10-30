from itertools import combinations

class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        if numFriends == 1:
            return word

        max_lex_largest_in_split = ""

        if numFriends == n:
            return max(list(word))

        for cut_indices_tuple in combinations(range(1, n), numFriends - 1):
            cut_indices = sorted(list(cut_indices_tuple))
            split_words = []
            start = 0
            for cut_index in cut_indices:
                split_words.append(word[start:cut_index])
                start = cut_index
            split_words.append(word[start:n])

            max_in_split = ""
            for sub in split_words:
                if max_in_split < sub:
                    max_in_split = sub

            if max_lex_largest_in_split < max_in_split:
                max_lex_largest_in_split = max_in_split

        return max_lex_largest_in_split