class Solution:
    def getSmallestString(self, s: str) -> str:
        s_list = list(s)
        current_min = s
        for i in range(len(s_list) - 1):
            a, b = s_list[i], s_list[i+1]
            if (int(a) % 2) == (int(b) % 2):
                temp = s_list.copy()
                temp[i], temp[i+1] = temp[i+1], temp[i]
                new_str = ''.join(temp)
                if new_str < current_min:
                    current_min = new_str
        return current_min