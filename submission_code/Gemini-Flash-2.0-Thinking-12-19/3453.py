class Solution:
    def validStrings(self, n: int) -> List[str]:
        memo = {}
        def generate_valid_strings(index, previous_char):
            if index == n:
                return [""]
            if (index, previous_char) in memo:
                return memo[(index, previous_char)]
            valid_strings = []
            if previous_char == '0':
                next_strings = generate_valid_strings(index + 1, '1')
                for s in next_strings:
                    valid_strings.append('1' + s)
            else: # previous_char == '1' or '2' (start)
                strings_with_0 = generate_valid_strings(index + 1, '0')
                for s in strings_with_0:
                    valid_strings.append('0' + s)
                strings_with_1 = generate_valid_strings(index + 1, '1')
                for s in strings_with_1:
                    valid_strings.append('1' + s)
            memo[(index, previous_char)] = valid_strings
            return valid_strings
            
        return generate_valid_strings(0, '2')