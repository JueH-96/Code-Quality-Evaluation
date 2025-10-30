from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        
        for i in range(len(s)):
            current_char = s[i]
            target_char = t[i]
            
            if current_char == target_char:
                continue
            
            # Calculate the index in the alphabet (0 for 'a', 1 for 'b', ..., 25 for 'z')
            current_index = ord(current_char) - ord('a')
            target_index = ord(target_char) - ord('a')
            
            # Calculate the forward and backward costs
            if target_index >= current_index:
                forward_steps = target_index - current_index
                backward_steps = 26 - forward_steps
            else:
                backward_steps = current_index - target_index
                forward_steps = 26 - backward_steps
            
            forward_cost = forward_steps * nextCost[current_index]
            backward_cost = backward_steps * previousCost[current_index]
            
            # Add the minimum cost to the total cost
            total_cost += min(forward_cost, backward_cost)
        
        return total_cost