class Solution:
    def getSmallestString(self, s: str) -> str:
        s_list = list(s)
        for i in range(len(s_list) - 1):
            a = int(s_list[i])
            b = int(s_list[i + 1])
            if (a % 2) == (b % 2):
                if a > b:
                    s_list[i], s_list[i + 1] = s_list[i + 1], s_list[i]
                    return ''.join(s_list)
        return ''.join(s_list)