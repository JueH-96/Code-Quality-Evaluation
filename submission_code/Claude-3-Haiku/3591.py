class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        n = len(s)
        total_cost = 0
        
        for i in range(n):
            s_char = s[i]
            t_char = t[i]
            
            if s_char == t_char:
                continue
            
            s_index = ord(s_char) - ord('a')
            t_index = ord(t_char) - ord('a')
            
            if t_index > s_index:
                total_cost += sum(nextCost[s_index:t_index])
            else:
                total_cost += sum(previousCost[t_index:s_index])
        
        return total_cost