class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        length = n + m - 1
        res = ['a'] * length

        def check(s):
            for i in range(n):
                if str1[i] == 'T':
                    if i + m > len(s) or s[i:i + m] != str2:
                        return False
                else:
                    if i + m <= len(s) and s[i:i + m] == str2:
                        return False
            return True

        def solve(index):
            if index == length:
                s = "".join(res)
                if check(s):
                    return s
                else:
                    return ""

            for char_code in range(ord('a'), ord('z') + 1):
                res[index] = chr(char_code)
                temp_res = solve(index + 1)
                if temp_res != "":
                    return temp_res
            return ""

        s = solve(0)
        return s