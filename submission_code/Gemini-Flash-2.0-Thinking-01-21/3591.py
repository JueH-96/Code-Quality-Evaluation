class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_distance = 0
        for i in range(len(s)):
            start_char = s[i]
            target_char = t[i]
            start_index = ord(start_char) - ord('a')
            target_index = ord(target_char) - ord('a')
            
            forward_shifts = (target_index - start_index + 26) % 26
            backward_shifts = (start_index - target_index + 26) % 26
            
            cost_forward = forward_shifts * nextCost[start_index]
            cost_backward = backward_shifts * previousCost[start_index]
            
            min_cost_i = min(cost_forward, cost_backward)
            total_distance += min_cost_i
            
        return total_distance