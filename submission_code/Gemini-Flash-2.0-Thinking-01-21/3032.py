class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_f_x = 0
        for start_player in range(n):
            current_player = start_player
            current_sum = start_player
            receivers_sequence = []
            player_sequence = [start_player]
            cycle_detected = False
            for _ in range(k):
                next_player = receiver[current_player]
                if next_player in player_sequence:
                    cycle_detected = True
                    cycle_start_index = player_sequence.index(next_player)
                    cycle_players = player_sequence[cycle_start_index:]
                    cycle_receivers = []
                    for p in cycle_players:
                        cycle_receivers.append(receiver[p])
                    cycle_receiver_sum = sum(cycle_receivers)
                    cycle_len = len(cycle_receivers)
                    remaining_passes = k - len(receivers_sequence)
                    num_cycles = remaining_passes // cycle_len
                    remaining_passes_in_cycle = remaining_passes % cycle_len
                    current_sum += sum(receivers_sequence) + num_cycles * cycle_receiver_sum
                    for j in range(remaining_passes_in_cycle):
                        current_sum += cycle_receivers[j]
                    break
                else:
                    receivers_sequence.append(next_player)
                    current_player = next_player
                    player_sequence.append(current_player)
            if not cycle_detected:
                current_sum += sum(receivers_sequence)
            max_f_x = max(max_f_x, current_sum)
        return max_f_x