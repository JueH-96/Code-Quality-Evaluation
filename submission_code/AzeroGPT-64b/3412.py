class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        n = len(s)
        s_dict = {}
        t_dict = {}
        for i in range(n):
            if s[i] in s_dict:
                s_dict[s[i]].append(i)
            else:
                s_dict[s[i]] = [i]
            if t[i] in t_dict:
                t_dict[t[i]].append(i)
            else:
                t_dict[t[i]] = [i]
        distance = 0
        for char in s:
            distance += abs(s_dict[char][0] - t_dict[char][0])
        return distance