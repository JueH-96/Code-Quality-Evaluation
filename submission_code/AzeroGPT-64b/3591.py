class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        cost = 0
        for i in range(len(s)):
            diff = (ord(t[i]) - ord(s[i])) % 26
            if diff == 0:
                continue
            elif diff < 13:
                cost += nextCost[ord(s[i]) - ord("a")] * diff
            else:
                cost += previousCost[ord(s[i]) - ord("a")] * (26 - diff)
        return cost