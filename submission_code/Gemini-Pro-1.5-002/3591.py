class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        ans = 0
        for i in range(len(s)):
            diff = ord(t[i]) - ord(s[i])
            if diff > 0:
                cost = nextCost[ord(s[i]) - ord('a')]
                ans += diff * cost
            elif diff < 0:
                cost = previousCost[ord(s[i]) - ord('a')]
                ans += abs(diff) * cost

        return ans