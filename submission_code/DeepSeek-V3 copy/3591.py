class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        for i in range(len(s)):
            s_char = s[i]
            t_char = t[i]
            if s_char == t_char:
                continue
            s_idx = ord(s_char) - ord('a')
            t_idx = ord(t_char) - ord('a')
            # Calculate the difference in both directions
            # Forward shift (next)
            forward_diff = (t_idx - s_idx) % 26
            forward_cost = forward_diff * nextCost[s_idx]
            # Backward shift (previous)
            backward_diff = (s_idx - t_idx) % 26
            backward_cost = backward_diff * previousCost[s_idx]
            # Choose the minimum cost
            total_cost += min(forward_cost, backward_cost)
        return total_cost