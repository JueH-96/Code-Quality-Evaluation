class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Precompute forward_cost and backward_cost arrays
        forward_cost = [[0] * 26 for _ in range(26)]
        for c in range(26):
            current_sum = 0
            for k in range(1, 26):
                current_char = (c + k - 1) % 26
                current_sum += nextCost[current_char]
                forward_cost[c][k] = current_sum
        
        backward_cost = [[0] * 26 for _ in range(26)]
        for c in range(26):
            current_sum = 0
            for k in range(1, 26):
                current_char = (c - k + 1) % 26
                current_sum += previousCost[current_char]
                backward_cost[c][k] = current_sum
        
        total_cost = 0
        for i in range(len(s)):
            s_char = ord(s[i]) - ord('a')
            t_char = ord(t[i]) - ord('a')
            if s_char == t_char:
                continue
            # Calculate forward and backward steps
            forward_steps = (t_char - s_char) % 26
            backward_steps = (s_char - t_char) % 26
            # Get costs
            cost_forward = forward_cost[s_char][forward_steps]
            cost_backward = backward_cost[s_char][backward_steps]
            total_cost += min(cost_forward, cost_backward)
        
        return total_cost