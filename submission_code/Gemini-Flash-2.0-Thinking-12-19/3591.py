class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        for i in range(len(s)):
            s_char = s[i]
            t_char = t[i]
            if s_char == t_char:
                continue
            start_index = ord(s_char) - ord('a')
            target_index = ord(t_char) - ord('a')
            forward_steps = (target_index - start_index) % 26
            backward_steps = (start_index - target_index) % 26
            forward_cost = 0
            current_char_index = start_index
            for _ in range(forward_steps):
                forward_cost += nextCost[current_char_index]
                current_char_index = (current_char_index + 1) % 26
            backward_cost = 0
            current_char_index = start_index
            for _ in range(backward_steps):
                backward_cost += previousCost[current_char_index]
                current_char_index = (current_char_index - 1 + 26) % 26
            total_cost += min(forward_cost, backward_cost)
        return total_cost