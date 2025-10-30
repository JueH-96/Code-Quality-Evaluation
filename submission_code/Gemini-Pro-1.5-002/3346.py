class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        s_list = list(s)
        for i in range(n):
            diff = ord(s_list[i]) - ord('a')
            if k >= diff:
                k -= diff
                s_list[i] = 'a'
            else:
                break
        
        for i in range(n - 1, -1, -1):
            if k == 0:
                break
            diff = ord(s_list[i]) - ord('a')
            if k > 0:
                change = min(k, 25 - diff)
                s_list[i] = chr(ord(s_list[i]) + change)
                k -= change

        return "".join(s_list)