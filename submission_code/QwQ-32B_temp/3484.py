class Solution:
    def getSmallestString(self, s: str) -> str:
        candidates = [s]
        s_list = list(s)
        for i in range(len(s_list) - 1):
            a = int(s_list[i])
            b = int(s_list[i+1])
            if (a % 2) == (b % 2):
                temp = s_list.copy()
                temp[i], temp[i+1] = temp[i+1], temp[i]
                candidates.append(''.join(temp))
        return min(candidates)