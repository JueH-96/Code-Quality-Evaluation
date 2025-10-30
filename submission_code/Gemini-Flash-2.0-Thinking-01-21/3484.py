class Solution:
    def getSmallestString(self, s: str) -> str:
        s_list = list(s)
        swapped = False
        for i in range(len(s) - 1):
            digit1 = int(s_list[i])
            digit2 = int(s_list[i+1])
            if (digit1 % 2) == (digit2 % 2):
                if digit1 > digit2 and not swapped:
                    s_list[i], s_list[i+1] = s_list[i+1], s_list[i]
                    swapped = True
                    break
        return "".join(s_list)