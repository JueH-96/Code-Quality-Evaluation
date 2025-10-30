class Solution:
    def smallestString(self, s: str) -> str:
        s_list = list(s)
        n = len(s)
        start = 0
        while start < n and s_list[start] == 'a':
            start += 1
        if start == n:
            # All characters are 'a', change the last one to 'z'
            s_list[-1] = 'z'
            return ''.join(s_list)
        # Find the end of the substring to transform
        end = start
        while end + 1 < n and s_list[end + 1] != 'a':
            end += 1
        # Transform each character in the substring
        for i in range(start, end + 1):
            s_list[i] = chr(ord(s_list[i]) - 1)
        return ''.join(s_list)