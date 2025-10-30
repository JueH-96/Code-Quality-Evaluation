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
            # Calculate the difference
            diff = (t_idx - s_idx) % 26
            # Calculate the cost for next shifts
            next_shift_cost = 0
            current = s_idx
            for _ in range(diff):
                next_shift_cost += nextCost[current]
                current = (current + 1) % 26
            # Calculate the cost for previous shifts
            prev_shift_cost = 0
            current = s_idx
            for _ in range(26 - diff):
                prev_shift_cost += previousCost[current]
                current = (current - 1) % 26
            # Choose the minimum cost
            total_cost += min(next_shift_cost, prev_shift_cost)
        return total_cost