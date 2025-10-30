class Solution:
    def getSmallestString(self, s: str) -> str:
        s_list = list(s)
        n = len(s_list)
        for i in range(n - 1):
            digit1 = int(s_list[i])
            digit2 = int(s_list[i+1])
            if (digit1 % 2 == digit2 % 2):
                if digit1 > digit2:
                    s_list[i], s_list[i+1] = s_list[i+1], s_list[i]
                    return "".join(s_list)
        return "".join(s_list)