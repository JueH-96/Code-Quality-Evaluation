class Solution:
    def possibleStringCount(self, word: str) -> int:
        groups = []
        i = 0
        while i < len(word):
            char = word[i]
            j = i
            while j < len(word) and word[j] == char:
                j += 1
            groups.append(word[i:j])
            i = j
        
        possible_strings = set()
        possible_strings.add(word)
        
        for group_index in range(len(groups)):
            group = groups[group_index]
            if len(group) > 1:
                char = group[0]
                for reduced_length in range(1, len(group)):
                    reduced_group = char * reduced_length
                    temp_groups = list(groups)
                    temp_groups[group_index] = reduced_group
                    new_word = "".join(temp_groups)
                    possible_strings.add(new_word)
                    
        return len(possible_strings)