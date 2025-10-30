class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        smallest_s = "~"

        for i in range(n):
            for j in range(i, n):
                temp_s_list = list(s)
                for k in range(i, j + 1):
                    if temp_s_list[k] == 'a':
                        temp_s_list[k] = 'z'
                    else:
                        temp_s_list[k] = chr(ord(temp_s_list[k]) - 1)
                modified_s = "".join(temp_s_list)
                if modified_s < smallest_s:
                    smallest_s = modified_s

        return smallest_s