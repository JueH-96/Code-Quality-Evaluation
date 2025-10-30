class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_f = 0

        for start_node in range(n):
            current_node = start_node
            f_x = start_node
            received_nodes = []
            visited = {start_node: 0}

            for num_passes in range(k):
                next_receiver = receiver[current_node]

                if next_receiver in visited:
                    cycle_start_index = visited[next_receiver]
                    cycle = received_nodes[cycle_start_index:]
                    cycle_len = len(cycle)
                    cycle_sum = sum(cycle)

                    remaining_passes = k - len(received_nodes)
                    num_full_cycles = remaining_passes // cycle_len
                    remaining_in_cycle = remaining_passes % cycle_len

                    f_x += num_full_cycles * cycle_sum + sum(cycle[:remaining_in_cycle])
                    break
                else:
                    received_nodes.append(next_receiver)
                    f_x += next_receiver
                    visited[next_receiver] = len(received_nodes)
                    current_node = next_receiver
            max_f = max(max_f, f_x)

        return max_f