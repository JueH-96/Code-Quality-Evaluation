class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        for i in range(len(s)):
            current_char = s[i]
            target_char = t[i]
            if current_char == target_char:
                continue
            
            current_index = ord(current_char) - ord('a')
            target_index = ord(target_char) - ord('a')
            
            if target_index > current_index:
                # Calculate cost to move forward
                forward_cost = sum(nextCost[current_index:target_index])
                # Calculate cost to wrap around and move backward
                backward_cost = sum(previousCost[current_index+1:]) + sum(previousCost[:target_index])
                total_cost += min(forward_cost, backward_cost)
            else:
                # Calculate cost to move backward
                backward_cost = sum(previousCost[current_index:target_index:-1])
                # Calculate cost to wrap around and move forward
                forward_cost = sum(nextCost[current_index+1:]) + sum(nextCost[:target_index])
                total_cost += min(forward_cost, backward_cost)
        
        return total_cost