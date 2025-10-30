class Solution:
    def getSmallestString(self, s: str) -> str:
        candidates = [s]
        s_list = list(s)
        for i in range(len(s_list) - 1):
            a, b = s_list[i], s_list[i+1]
            if (int(a) % 2) == (int(b) % 2):
                new_list = s_list.copy()
                new_list[i], new_list[i+1] = new_list[i+1], new_list[i]
                candidates.append(''.join(new_list))
        return min(candidates)