class Solution:
    def getSmallestString(self, s: str) -> str:
        n = len(s)
        s_list = list(s)

        for i in range(n - 1):
            digit1 = int(s_list[i])
            digit2 = int(s_list[i+1])

            if (digit1 % 2 == digit2 % 2):
                if s_list[i] > s_list[i+1]:
                    s_list[i], s_list[i+1] = s_list[i+1], s_list[i]
                    return "".join(s_list)

        return "".join(s_list)