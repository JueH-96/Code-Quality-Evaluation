class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        current_s = s
        prev_s = s
        while current_s:
            prev_s = current_s
            next_s = ""
            removed_indices = {}
            indices_to_remove = []
            temp_s = list(current_s)
            for char_code in range(ord('a'), ord('z') + 1):
                char = chr(char_code)
                try:
                    index_to_remove = -1
                    for i in range(len(temp_s)):
                        if temp_s[i] == char:
                            index_to_remove = i
                            break
                    if index_to_remove != -1:
                        indices_to_remove.append(index_to_remove)
                except ValueError:
                    pass

            indices_to_remove.sort()
            next_s_list = []
            removed_count = 0
            current_index = 0
            for i in range(len(temp_s)):
                is_removed = False
                for remove_index in indices_to_remove:
                    if i == remove_index:
                        is_removed = True
                        break
                if not is_removed:
                    next_s_list.append(temp_s[i])
            current_s = "".join(next_s_list)
            if not current_s:
                return prev_s
        return prev_s