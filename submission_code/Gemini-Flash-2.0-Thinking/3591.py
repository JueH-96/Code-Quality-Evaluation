class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        n = len(s)

        for i in range(n):
            start_char = s[i]
            target_char = t[i]

            if start_char == target_char:
                continue

            start_ord = ord(start_char) - ord('a')
            target_ord = ord(target_char) - ord('a')

            forward_cost = 0
            current_ord = start_ord
            while current_ord != target_ord:
                forward_cost += nextCost[current_ord]
                current_ord = (current_ord + 1) % 26

            backward_cost = 0
            current_ord = start_ord
            while current_ord != target_ord:
                backward_cost += previousCost[current_ord]
                current_ord = (current_ord - 1 + 26) % 26

            total_cost += min(forward_cost, backward_cost)

        return total_cost