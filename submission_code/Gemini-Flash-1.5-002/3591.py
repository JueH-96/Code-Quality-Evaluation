class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        ans = 0
        for i in range(len(s)):
            diff = ord(t[i]) - ord(s[i])
            if diff == 0:
                continue
            elif diff > 0:
                cost1 = 0
                cur = ord(s[i])
                for _ in range(diff):
                    cost1 += nextCost[cur - ord('a')]
                    cur +=1
                
                cost2 = 0
                cur = ord(s[i])
                for _ in range(26 - diff):
                    cost2 += previousCost[cur - ord('a')]
                    cur -= 1
                
                ans += min(cost1, cost2)
            else:
                cost1 = 0
                cur = ord(s[i])
                for _ in range(-diff):
                    cost1 += previousCost[cur - ord('a')]
                    cur -= 1
                
                cost2 = 0
                cur = ord(s[i])
                for _ in range(26 + diff):
                    cost2 += nextCost[cur - ord('a')]
                    cur += 1
                ans += min(cost1, cost2)
        return ans