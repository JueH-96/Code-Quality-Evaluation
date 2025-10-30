class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        for i in range(len(s)):
            start_char = s[i]
            end_char = t[i]
            
            # Calculate the direct distance in the alphabet
            start_index = ord(start_char) - ord('a')
            end_index = ord(end_char) - ord('a')
            
            if start_index <= end_index:
                forward_steps = end_index - start_index
                backward_steps = 26 - forward_steps
            else:
                backward_steps = start_index - end_index
                forward_steps = 26 - backward_steps
            
            # Calculate the cost for both directions
            forward_cost = sum(nextCost[(start_index + j) % 26] for j in range(forward_steps))
            backward_cost = sum(previousCost[(start_index - j) % 26] for j in range(backward_steps))
            
            # Add the minimum cost to the total cost
            total_cost += min(forward_cost, backward_cost)
        
        return total_cost