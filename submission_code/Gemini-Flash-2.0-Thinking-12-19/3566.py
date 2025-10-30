class Solution:
    def stringSequence(self, target: str) -> List[str]:
        output_list = []
        current_string = ""
        for char in target:
            if not current_string:
                current_char_code = ord('a')
                target_char_code = ord(char)
                while current_char_code <= target_char_code:
                    current_string = chr(current_char_code)
                    output_list.append(current_string)
                    current_char_code += 1
            else:
                current_string += 'a'
                output_list.append(current_string)
                last_char_index = len(current_string) - 1
                current_last_char_code = ord('a')
                target_char_code = ord(char)
                while current_last_char_code < target_char_code:
                    current_last_char_code += 1
                    current_string = current_string[:-1] + chr(current_last_char_code)
                    output_list.append(current_string)
        return output_list