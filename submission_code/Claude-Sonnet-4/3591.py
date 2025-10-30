class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        
        for i in range(len(s)):
            if s[i] == t[i]:
                continue
                
            start = ord(s[i]) - ord('a')
            end = ord(t[i]) - ord('a')
            
            # Calculate cost going forward (next)
            forward_cost = 0
            curr = start
            while curr != end:
                forward_cost += nextCost[curr]
                curr = (curr + 1) % 26
            
            # Calculate cost going backward (previous)
            backward_cost = 0
            curr = start
            while curr != end:
                curr = (curr - 1) % 26
                backward_cost += previousCost[curr]
            
            # Take minimum cost
            total_cost += min(forward_cost, backward_cost)
        
        return total_cost