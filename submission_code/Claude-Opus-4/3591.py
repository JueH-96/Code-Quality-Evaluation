class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        
        for i in range(len(s)):
            # Get the character positions in alphabet (0-25)
            start = ord(s[i]) - ord('a')
            end = ord(t[i]) - ord('a')
            
            if start == end:
                # No shift needed
                continue
            
            # Calculate forward shift cost
            forward_cost = 0
            curr = start
            while curr != end:
                forward_cost += nextCost[curr]
                curr = (curr + 1) % 26
            
            # Calculate backward shift cost
            backward_cost = 0
            curr = start
            while curr != end:
                curr = (curr - 1) % 26
                backward_cost += previousCost[curr]
            
            # Choose minimum cost
            total_cost += min(forward_cost, backward_cost)
        
        return total_cost