class Solution:
    def getMaxFunctionValue(self, receiver: list[int], k: int) -> int:
        n = len(receiver)
        next_player = [0] * n
        cycle_length = [0] * n
        cycle_sum = [0] * n
        visited = [0] * n
        max_value = 0

        for i in range(n):
            j = i
            while visited[j] == 0:
                visited[j] = i + 1
                j = receiver[j]
            if visited[j] == i + 1:
                k = i
                while k != j:
                    cycle_length[j] += 1
                    cycle_sum[j] += k
                    k = receiver[k]
                cycle_length[j] += 1
                cycle_sum[j] += j
            next_player[i] = j

        for i in range(n):
            if k >= cycle_length[next_player[i]]:
                max_value = max(max_value, i + cycle_sum[next_player[i]] + (k - cycle_length[next_player[i]]) // cycle_length[next_player[i]] * cycle_sum[next_player[i]])
            else:
                j = i
                total = 0
                for _ in range(k + 1):
                    total += j
                    j = receiver[j]
                max_value = max(max_value, total)
        return max_value