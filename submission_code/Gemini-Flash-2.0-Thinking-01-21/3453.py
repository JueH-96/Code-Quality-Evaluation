class Solution:
    def validStrings(self, n: int) -> List[str]:
        def is_valid(s):
            for i in range(len(s) - 1):
                if s[i:i+2] == "00":
                    return False
            return True

        if n == 0:
            return [""]
        if n == 1:
            return ["0", "1"]

        valid_strings_list = []
        for i in range(2**n):
            binary_string = bin(i)[2:].zfill(n)
            if is_valid(binary_string):
                valid_strings_list.append(binary_string)
        return valid_strings_list