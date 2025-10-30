class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        visited = [False] * n
        in_cycle = [False] * n
        cycle_id = [0] * n
        path_sum = [0] * n
        path_length = [0] * n
        cycle_sum = {}
        cycle_length = {}
        cycle_nodes = {}
        cycle_prefix_sums = {}

        current_cycle_id = 0

        for x in range(n):
            if not visited[x]:
                path = []
                current = x
                while True:
                    if visited[current]:
                        if current in path:
                            # Found a cycle
                            cycle_start = path.index(current)
                            cycle = path[cycle_start:]
                            c_len = len(cycle)
                            c_sum = sum(cycle)
                            cycle_sum[current_cycle_id] = c_sum
                            cycle_length[current_cycle_id] = c_len
                            cycle_nodes[current_cycle_id] = cycle
                            for node in cycle:
                                in_cycle[node] = True
                                cycle_id[node] = current_cycle_id
                            # Assign path sums
                            for i in range(cycle_start):
                                node = path[i]
                                visited[node] = True
                                path_length[node] = i + 1
                                path_sum[node] = sum(path[:i+1])
                                cycle_id[node] = current_cycle_id
                            current_cycle_id += 1
                            break
                        else:
                            # Cycle already processed
                            # Assign path sums leading to this cycle
                            for i in range(len(path)):
                                node = path[i]
                                visited[node] = True
                                path_length[node] = i + 1
                                path_sum[node] = sum(path[:i+1])
                                cycle_id[node] = cycle_id[current]
                            break
                    else:
                        visited[current] = True
                        path.append(current)
                        current = receiver[current]

        # Compute cycle prefix sums
        for cid in range(current_cycle_id):
            cycle = cycle_nodes[cid]
            prefix = [0]
            for node in cycle:
                prefix.append(prefix[-1] + node)
            cycle_prefix_sums[cid] = prefix

        max_f = 0
        for x in range(n):
            if path_length[x] > k + 1:
                # Need to sum x and the first k receivers
                # Since path_length[x] > k+1, all steps stay in the path
                total = x
                current = x
                for _ in range(k):
                    current = receiver[current]
                    total += current
                max_f = max(max_f, total)
            else:
                # Sum x, path_sum[x], and the cycle sum
                total = x + path_sum[x]
                remaining = k - path_length[x]
                if remaining > 0:
                    cycles = remaining // cycle_length[cycle_id[x]]
                    total += cycles * cycle_sum[cycle_id[x]]
                    rem = remaining % cycle_length[cycle_id[x]]
                    total += cycle_prefix_sums[cycle_id[x]][rem]
                max_f = max(max_f, total)
        return max_f