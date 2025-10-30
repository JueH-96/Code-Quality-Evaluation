class Solution:
    def getSmallestString(self, s: str) -> str:
        min_str = s
        s_list = list(s)
        n = len(s)
        for i in range(n - 1):
            a = int(s[i])
            b = int(s[i + 1])
            if (a % 2) == (b % 2):
                new_list = s_list.copy()
                new_list[i], new_list[i + 1] = new_list[i + 1], new_list[i]
                new_str = ''.join(new_list)
                if new_str < min_str:
                    min_str = new_str
        return min_str