class Solution:
    def punishmentNumber(self, n: int) -> int:
        punishment_sum = 0

        def canPartition(square_str, target, memo):
            if (square_str, target) in memo:
                return memo[(square_str, target)]

            if not square_str:
                return target == 0

            res = False
            n_str = len(square_str)
            for i in range(1, n_str + 1):
                prefix = square_str[:i]
                remaining = square_str[i:]
                prefix_val = int(prefix)
                if prefix_val <= target and canPartition(remaining, target - prefix_val, memo):
                    res = True
                    break

            memo[(square_str, target)] = res
            return res

        for i in range(1, n + 1):
            square = i * i
            square_str = str(square)
            memo = {}
            if canPartition(square_str, i, memo):
                punishment_sum += square

        return punishment_sum