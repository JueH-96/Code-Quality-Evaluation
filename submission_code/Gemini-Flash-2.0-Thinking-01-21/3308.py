class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        current_s = list(s)
        previous_s = list(s)
        while current_s:
            previous_s = list(current_s)
            remove_indices = []
            seen_chars = set()
            for char_to_remove in 'abcdefghijklmnopqrstuvwxyz':
                first_index = -1
                for index in range(len(current_s)):
                    if current_s[index] == char_to_remove and char_to_remove not in seen_chars:
                        first_index = index
                        seen_chars.add(char_to_remove)
                        break
                if first_index != -1:
                    remove_indices.append(first_index)
            remove_indices.sort(reverse=True)
            for index in remove_indices:
                current_s.pop(index)
        return "".join(previous_s)