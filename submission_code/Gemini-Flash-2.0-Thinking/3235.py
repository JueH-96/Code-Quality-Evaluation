class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        n = len(source)
        if n != len(target):
            return -1

        char_map = {chr(ord('a') + i): i for i in range(26)}
        cost_matrix = [[float('inf')] * 26 for _ in range(26)]

        for i in range(26):
            cost_matrix[i][i] = 0

        for i in range(len(original)):
            orig_char = original[i]
            changed_char = changed[i]
            c = cost[i]
            cost_matrix[char_map[orig_char]][char_map[changed_char]] = min(
                cost_matrix[char_map[orig_char]][char_map[changed_char]], c
            )

        for k in range(26):
            for i in range(26):
                for j in range(26):
                    cost_matrix[i][j] = min(cost_matrix[i][j], cost_matrix[i][k] + cost_matrix[k][j])

        total_cost = 0
        for i in range(n):
            if source[i] != target[i]:
                source_char_index = char_map[source[i]]
                target_char_index = char_map[target[i]]
                change_cost = cost_matrix[source_char_index][target_char_index]
                if change_cost == float('inf'):
                    return -1
                total_cost += change_cost

        return total_cost