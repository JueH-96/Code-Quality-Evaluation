class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        for i in range(len(s)):
            char_s = ord(s[i]) - ord('a')
            char_t = ord(t[i]) - ord('a')
            
            diff = char_t - char_s
            if diff == 0:
                continue
            
            if diff > 0:
                cost1 = 0
                for j in range(diff):
                    cost1 += nextCost[(char_s + j) % 26]
                
                cost2 = 0
                for j in range(26 - diff):
                    cost2 += previousCost[(char_s + diff + j) % 26]
                
                total_cost += min(cost1, cost2)
            else:
                diff = -diff
                cost1 = 0
                for j in range(diff):
                    cost1 += previousCost[(char_s - j - 1 + 26) % 26]
                
                cost2 = 0
                for j in range(26 - diff):
                    cost2 += nextCost[(char_s - diff - j - 1 + 26) % 26]
                
                total_cost += min(cost1, cost2)
        
        return total_cost