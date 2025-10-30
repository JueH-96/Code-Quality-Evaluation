class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        # Precompute the power of 2 jumps
        max_power = 30  # since 2^30 > 10^9
        jump = [[0] * n for _ in range(max_power + 1)]
        value = [[0] * n for _ in range(max_power + 1)]

        for i in range(n):
            jump[0][i] = receiver[i]
            value[0][i] = receiver[i]

        for p in range(1, max_power + 1):
            for i in range(n):
                jump[p][i] = jump[p - 1][jump[p - 1][i]]
                value[p][i] = value[p - 1][i] + value[p - 1][jump[p - 1][i]]

        max_value = 0
        for start in range(n):
            current = start
            current_value = start
            for p in range(max_power + 1):
                if (k >> p) & 1:
                    current_value += value[p][current]
                    current = jump[p][current]
            max_value = max(max_value, current_value)

        return max_value