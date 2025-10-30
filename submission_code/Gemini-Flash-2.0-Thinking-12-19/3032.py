class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_f_x = -1
        for start_player in range(n):
            current_player = start_player
            total_sum = start_player
            receivers_path = []
            visited_receivers_index = {}
            cycle_detected = False
            for pass_num in range(k):
                next_receiver = receiver[current_player]
                if next_receiver in visited_receivers_index:
                    cycle_start_index = visited_receivers_index[next_receiver]
                    cycle = receivers_path[cycle_start_index:]
                    cycle_sum = sum(cycle)
                    cycle_len = len(cycle)
                    remaining_passes = k - pass_num
                    num_cycles = remaining_passes // cycle_len
                    total_sum += num_cycles * cycle_sum
                    remaining_in_cycle_passes = remaining_passes % cycle_len
                    for i in range(remaining_in_cycle_passes):
                        total_sum += cycle[i]
                    total_sum += next_receiver
                    cycle_detected = True
                    break
                else:
                    total_sum += next_receiver
                    visited_receivers_index[next_receiver] = pass_num
                    receivers_path.append(next_receiver)
                    current_player = next_receiver
            if not cycle_detected:
                pass # total_sum is already calculated.
            max_f_x = max(max_f_x, total_sum)
        return max_f_x