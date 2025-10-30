class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        
        for i in range(len(s)):
            src = ord(s[i]) - ord('a')
            dst = ord(t[i]) - ord('a')
            
            if src == dst:
                continue
            
            # Calculate forward cost
            forward_cost = 0
            curr = src
            while curr != dst:
                forward_cost += nextCost[curr]
                curr = (curr + 1) % 26
            
            # Calculate backward cost
            backward_cost = 0
            curr = src
            while curr != dst:
                backward_cost += previousCost[curr]
                curr = (curr - 1 + 26) % 26
            
            total_cost += min(forward_cost, backward_cost)
        
        return total_cost