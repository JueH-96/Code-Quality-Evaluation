from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_value = 0

        for start in range(n):
            current = start
            total = 0
            visited = {}

            for _ in range(k):
                if current in visited:
                    cycle_length = _ - visited[current]
                    cycles_remaining = (k - _) // cycle_length
                    cycle_sum = total - visited[current][1]
                    total += cycle_sum * cycles_remaining
                    _ += cycle_length * cycles_remaining
                    visited[current] = (_, total)
                else:
                    visited[current] = (_, total)

                total += current
                current = receiver[current]

            max_value = max(max_value, total)

        return max_value