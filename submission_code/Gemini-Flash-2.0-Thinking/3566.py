class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current_string = ""

        for i in range(len(target)):
            if i == 0:
                current_string = "a"
                result.append(current_string)
                while current_string != target[0]:
                    last_char = current_string[-1]
                    current_string = current_string[:-1] + self.next_char(last_char)
                    result.append(current_string)
            else:
                # Current string should be target[:i]
                current_string = target[:i] + "a"
                result.append(current_string)
                while current_string[-1] != target[i]:
                    last_char = current_string[-1]
                    current_string = current_string[:-1] + self.next_char(last_char)
                    result.append(current_string)

        return result

    def next_char(self, char):
        if char == 'z':
            return 'a'
        else:
            return chr(ord(char) + 1)