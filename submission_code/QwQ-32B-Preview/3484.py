class Solution:
    def getSmallestString(self, s: str) -> str:
        smallest = s
        s_list = list(s)
        for i in range(len(s_list) - 1):
            if int(s_list[i]) % 2 == int(s_list[i+1]) % 2:
                s_list[i], s_list[i+1] = s_list[i+1], s_list[i]
                new_s = ''.join(s_list)
                if new_s < smallest:
                    smallest = new_s
                s_list[i], s_list[i+1] = s_list[i+1], s_list[i]  # swap back
        return smallest