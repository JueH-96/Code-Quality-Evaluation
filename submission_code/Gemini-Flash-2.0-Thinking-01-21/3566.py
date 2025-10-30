class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current_string = ""
        for target_char in target:
            if not current_string:
                current_string += 'a'
                result.append(current_string)
                while current_string[-1] != target_char:
                    last_char = current_string[-1]
                    next_char = chr(ord(last_char) + 1)
                    if next_char > 'z':
                        next_char = 'a'
                    current_string = current_string[:-1] + next_char
                    result.append(current_string)
            else:
                current_string += 'a'
                result.append(current_string)
                while current_string[-1] != target_char:
                    last_char = current_string[-1]
                    next_char = chr(ord(last_char) + 1)
                    if next_char > 'z':
                        next_char = 'a'
                    current_string = current_string[:-1] + next_char
                    result.append(current_string)
        return result