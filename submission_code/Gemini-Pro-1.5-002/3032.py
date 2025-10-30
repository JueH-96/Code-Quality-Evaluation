class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        next_receiver = [0] * n
        for i in range(n):
            next_receiver[i] = receiver[i]

        paths = []
        for start_node in range(n):
            path = [start_node]
            curr = start_node
            for _ in range(min(n, k)):
                curr = next_receiver[curr]
                path.append(curr)
            paths.append(path)

        max_f = 0
        for start_node in range(n):
            f = 0
            path = paths[start_node]
            for i in range(min(k + 1, len(path))):
                f += path[i]

            if k + 1 > len(path):
                cycle_start_index = 0
                for i in range(len(path)):
                    if path[i] in path[i+1:]:
                        cycle_start_index = i
                        break
                
                cycle = path[cycle_start_index:]
                cycle_sum = sum(cycle)
                remaining_passes = k + 1 - len(path)
                num_cycles = remaining_passes // len(cycle)
                f += num_cycles * cycle_sum
                remaining_passes %= len(cycle)
                for i in range(remaining_passes):
                    f += cycle[i]

            max_f = max(max_f, f)

        return max_f