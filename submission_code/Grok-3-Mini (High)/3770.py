class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        len_word = n + m - 1
        word = [None] * len_word
        is_fixed = [False] * len_word
        
        # Set the 'T' constraints
        for i in range(n):
            if str1[i] == 'T':
                for p in range(m):
                    pos = i + p
                    if word[pos] is not None:
                        if word[pos] != str2[p]:
                            return ""
                    else:
                        word[pos] = str2[p]
                        is_fixed[pos] = True
        
        # Set unset positions to 'a'
        for idx in range(len_word):
            if word[idx] is None:
                word[idx] = 'a'
        
        # Handle 'F' constraints from left to right
        for i in range(n):
            if str1[i] == 'F':
                # Check if the substring equals str2
                if all(word[i + k] == str2[k] for k in range(m)):
                    # Need to make it unequal, find rightmost non-fixed position
                    change_pos = -1
                    for k in range(m - 1, -1, -1):
                        p = i + k
                        if not is_fixed[p]:
                            change_pos = p
                            break
                    if change_pos == -1:
                        return ""  # No position to change, impossible
                    # Set to smallest char != str2[change_pos - i]
                    target_char = str2[change_pos - i]
                    if target_char > 'a':
                        new_char = 'a'
                    else:
                        new_char = 'b'
                    word[change_pos] = new_char
        
        # Join the word list and return
        return ''.join(word)