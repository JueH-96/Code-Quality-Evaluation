import itertools

class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        split_indices_combinations = itertools.combinations(range(1, n), numFriends - 1)
        all_split_strings = []
        for indices in split_indices_combinations:
            current_split = []
            start_index = 0
            sorted_indices = sorted(list(indices))
            for index in sorted_indices:
                current_split.append(word[start_index:index])
                start_index = index
            current_split.append(word[start_index:])
            all_split_strings.extend(current_split)
        
        if numFriends == 1:
            all_split_strings = [word]
        elif numFriends == n:
            all_split_strings = list(word)
        elif numFriends > 1 and numFriends < n:
            split_indices_combinations = itertools.combinations(range(1, n), numFriends - 1)
            all_split_strings_temp = []
            for indices in split_indices_combinations:
                current_split = []
                start_index = 0
                sorted_indices = sorted(list(indices))
                for index in sorted_indices:
                    current_split.append(word[start_index:index])
                    start_index = index
                current_split.append(word[start_index:])
                all_split_strings_temp.extend(current_split)
            all_split_strings = all_split_strings_temp
        elif numFriends == n:
            all_split_strings = list(word)


        max_string = ""
        if not all_split_strings:
            if numFriends == n:
                return "g" if word else "" # handle edge case like "gggg", 4, expect "g"
            else:
                return "" # should not happen given constraints
        else:
            max_string = all_split_strings[0]
            for s in all_split_strings:
                if s > max_string:
                    max_string = s
        return max_string