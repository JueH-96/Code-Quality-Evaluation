class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        
        for i in range(len(s)):
            src = ord(s[i]) - ord('a')
            dst = ord(t[i]) - ord('a')
            
            if src == dst:
                continue
            
            # Calculate cost going forward (next operations)
            forward_cost = 0
            current = src
            while current != dst:
                forward_cost += nextCost[current]
                current = (current + 1) % 26
            
            # Calculate cost going backward (previous operations)
            backward_cost = 0
            current = src
            while current != dst:
                backward_cost += previousCost[current]
                current = (current - 1) % 26
            
            # Choose the minimum cost
            total_cost += min(forward_cost, backward_cost)
        
        return total_cost