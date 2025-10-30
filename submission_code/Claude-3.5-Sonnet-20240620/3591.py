class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        
        for i in range(len(s)):
            s_char, t_char = ord(s[i]) - ord('a'), ord(t[i]) - ord('a')
            
            if s_char == t_char:
                continue
            
            forward_cost = 0
            backward_cost = 0
            
            if s_char < t_char:
                forward_dist = t_char - s_char
                backward_dist = 26 - forward_dist
                
                for j in range(forward_dist):
                    forward_cost += nextCost[(s_char + j) % 26]
                
                for j in range(backward_dist):
                    backward_cost += previousCost[(s_char - j - 1 + 26) % 26]
            else:
                backward_dist = s_char - t_char
                forward_dist = 26 - backward_dist
                
                for j in range(forward_dist):
                    forward_cost += nextCost[(s_char + j) % 26]
                
                for j in range(backward_dist):
                    backward_cost += previousCost[(s_char - j - 1 + 26) % 26]
            
            total_cost += min(forward_cost, backward_cost)
        
        return total_cost