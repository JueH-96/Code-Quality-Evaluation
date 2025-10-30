class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        def cost_to_shift(from_char, to_char):
            from_index = ord(from_char) - ord('a')
            to_index = ord(to_char) - ord('a')
            forward_cost = (to_index - from_index) % 26
            backward_cost = (from_index - to_index) % 26
            return min(forward_cost * nextCost[from_index], backward_cost * previousCost[from_index])

        total_cost = 0
        for i in range(len(s)):
            total_cost += cost_to_shift(s[i], t[i])

        return total_cost