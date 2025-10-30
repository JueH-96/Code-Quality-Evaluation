class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        res = []
        current_budget = k
        for char in s:
            for c in "abcdefghijklmnopqrstuvwxyz":
                diff = abs(ord(char) - ord(c))
                cost = min(diff, 26 - diff)
                if cost <= current_budget:
                    res.append(c)
                    current_budget -= cost
                    break
        return "".join(res)