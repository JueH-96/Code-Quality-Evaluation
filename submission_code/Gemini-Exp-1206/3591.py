class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        ans = 0
        for i in range(len(s)):
            if s[i] == t[i]:
                continue
            s_index = ord(s[i]) - ord('a')
            t_index = ord(t[i]) - ord('a')
            diff = t_index - s_index
            if diff > 0:
                ans += min(diff * nextCost[s_index], (26 - diff) * previousCost[s_index])
            else:
                diff = abs(diff)
                ans += min(diff * previousCost[s_index], (26 - diff) * nextCost[s_index])
        return ans