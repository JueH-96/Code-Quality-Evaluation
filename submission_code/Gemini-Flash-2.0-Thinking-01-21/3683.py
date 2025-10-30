import itertools

class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        split_positions_indices = list(range(1, n))
        max_lex_string = ""

        if numFriends == 1:
            return word
        if numFriends == n:
            for char in word:
                if char > max_lex_string:
                    max_lex_string = char
            return max_lex_string

        if numFriends > n:
            return "" # Should not happen based on constraints, but just in case.

        num_splits_needed = numFriends - 1
        if num_splits_needed > len(split_positions_indices):
            split_positions_combinations = [[]] # Only one possible split if numFriends == 1
        else:
            split_positions_combinations = list(itertools.combinations(split_positions_indices, num_splits_needed))

        for split_positions in split_positions_combinations:
            current_splits = []
            last_split_index = 0
            split_positions_list = sorted(list(split_positions))
            for split_pos in split_positions_list:
                current_splits.append(word[last_split_index:split_pos])
                last_split_index = split_pos
            current_splits.append(word[last_split_index:])

            for s in current_splits:
                if s > max_lex_string:
                    max_lex_string = s
        
        if not split_positions_combinations: # For cases where numFriends == 1 or numFriends == n
            if numFriends == 1:
                return word
            elif numFriends == n:
                for char in word:
                    if char > max_lex_string:
                        max_lex_string = char
                return max_lex_string

        return max_lex_string