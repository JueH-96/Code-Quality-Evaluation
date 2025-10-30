class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        def get_min_cost(from_char: str, to_char: str) -> int:
            # Get indices in alphabet (0-25)
            from_idx = ord(from_char) - ord('a')
            to_idx = ord(to_char) - ord('a')
            
            # Calculate forward distance
            forward_steps = (to_idx - from_idx) % 26
            # Calculate backward distance
            backward_steps = (from_idx - to_idx) % 26
            
            # Calculate forward cost
            forward_cost = 0
            curr_idx = from_idx
            for _ in range(forward_steps):
                forward_cost += nextCost[curr_idx]
                curr_idx = (curr_idx + 1) % 26
            
            # Calculate backward cost
            backward_cost = 0
            curr_idx = from_idx
            for _ in range(backward_steps):
                backward_cost += previousCost[curr_idx]
                curr_idx = (curr_idx - 1) % 26
            
            # Return minimum of forward and backward costs
            return min(forward_cost, backward_cost)
        
        total_cost = 0
        # For each character position
        for i in range(len(s)):
            if s[i] != t[i]:
                # Add minimum cost to transform s[i] to t[i]
                total_cost += get_min_cost(s[i], t[i])
        
        return total_cost