from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        in_cycle = [False] * n
        cycle_sum = [0] * n
        cycle_length = [0] * n
        cycle_list = {}
        prefix_sums = {}

        # Iterative DFS for cycle detection
        for i in range(n):
            if in_cycle[i]:
                continue
            visited = set()
            path = []
            stack = [i]
            while stack:
                u = stack[-1]
                if u in visited:
                    # Cycle detected
                    cycle_start = stack.index(u)
                    cycle_nodes = stack[cycle_start:]
                    cycle_sum_u = sum(cycle_nodes)
                    cycle_length_u = len(cycle_nodes)
                    for node in cycle_nodes:
                        in_cycle[node] = True
                        cycle_sum[node] = cycle_sum_u
                        cycle_length[node] = cycle_length_u
                    cycle_list[i] = cycle_nodes
                    prefix_sums[i] = [0] * (cycle_length_u + 1)
                    for idx in range(cycle_length_u):
                        prefix_sums[i][idx + 1] = prefix_sums[i][idx] + cycle_nodes[idx]
                    stack.pop()
                    break
                visited.add(u)
                path.append(u)
                v = receiver[u]
                if v in stack:
                    # Cycle detected
                    cycle_start = stack.index(v)
                    cycle_nodes = stack[cycle_start:] + [v]
                    cycle_sum_u = sum(cycle_nodes)
                    cycle_length_u = len(cycle_nodes)
                    for node in cycle_nodes:
                        in_cycle[node] = True
                        cycle_sum[node] = cycle_sum_u
                        cycle_length[node] = cycle_length_u
                    cycle_list[i] = cycle_nodes
                    prefix_sums[i] = [0] * (cycle_length_u + 1)
                    for idx in range(cycle_length_u):
                        prefix_sums[i][idx + 1] = prefix_sums[i][idx] + cycle_nodes[idx]
                    stack.pop()
                    break
                stack.append(v)
            path.clear()

        # BFS to compute distance and sum to cycle
        from collections import deque
        distance_to_cycle = [-1] * n
        path_sum_to_cycle = [-1] * n
        queue = deque()
        for u in range(n):
            if in_cycle[u]:
                distance_to_cycle[u] = 0
                path_sum_to_cycle[u] = 0  # Excluding u itself
                queue.append(u)
        while queue:
            u = queue.popleft()
            v = receiver[u]
            if distance_to_cycle[v] == -1:
                distance_to_cycle[v] = distance_to_cycle[u] + 1
                path_sum_to_cycle[v] = path_sum_to_cycle[u] + v
                queue.append(v)

        # Compute f(x) for each x
        max_sum = 0
        for x in range(n):
            if in_cycle[x]:
                # x is in cycle
                cycle = cycle_list[x]
                cycle_pos = cycle.index(x)
                full_cycles = (k - 1) // cycle_length[x]
                remainder = (k - 1) % cycle_length[x]
                current_sum = x + full_cycles * (cycle_sum[x] - x) + (prefix_sums[x][cycle_pos + remainder] - prefix_sums[x][cycle_pos])
                max_sum = max(max_sum, current_sum)
            else:
                # x is not in cycle
                l = distance_to_cycle[x]
                if k < l:
                    # Sum the first k+1 terms along the path to the cycle
                    sum_path = x
                    current = x
                    for _ in range(k):
                        sum_path += receiver[current]
                        current = receiver[current]
                    max_sum = max(max_sum, sum_path)
                else:
                    # Sum = x + path_sum_to_cycle[x] + full cycles + remainder
                    sum_path = x + path_sum_to_cycle[x]
                    full_cycles = (k - l) // cycle_length[x]
                    remainder = (k - l) % cycle_length[x]
                    sum_cycles = full_cycles * cycle_sum[x]
                    sum_remainder = prefix_sums[x][remainder]
                    current_sum = sum_path + sum_cycles + sum_remainder
                    max_sum = max(max_sum, current_sum)
        
        return max_sum