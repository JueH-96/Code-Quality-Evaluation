class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        
        for i in range(len(s)):
            # Calculate forward distance (s[i] -> t[i])
            forward_distance = (ord(t[i]) - ord(s[i])) % 26
            # Calculate backward distance (s[i] -> t[i])
            backward_distance = (26 - forward_distance) % 26
            
            # Calculate cost of moving forward
            forward_cost = 0
            current = ord(s[i]) - ord('a')
            for _ in range(forward_distance):
                forward_cost += nextCost[current]
                current = (current + 1) % 26
            
            # Calculate cost of moving backward
            backward_cost = 0
            current = ord(s[i]) - ord('a')
            for _ in range(backward_distance):
                backward_cost += previousCost[current]
                current = (current - 1) % 26
            
            # Choose the more efficient transformation
            total_cost += min(forward_cost, backward_cost)
        
        return total_cost