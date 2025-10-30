import collections

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        min_total_moves = float('inf')
        for alice_index in range(n):
            initial_nums = list(nums)
            initial_picked_count = 0
            if initial_nums[alice_index] == 1:
                initial_picked_count = 1
                initial_nums[alice_index] = 0
            if initial_picked_count >= k:
                min_total_moves = min(min_total_moves, 0)
                continue
            
            start_state = (tuple(initial_nums), 0, initial_picked_count, 0) # (current_nums, changes_made, ones_picked, moves)
            queue = collections.deque([start_state])
            visited_states = {tuple(initial_nums)}
            min_moves_for_index = float('inf')
            
            while queue:
                current_nums_tuple, changes_made, ones_picked, moves = queue.popleft()
                current_nums = list(current_nums_tuple)
                if ones_picked >= k:
                    min_moves_for_index = min(min_moves_for_index, moves)
                    continue
                    
                # Operation 1: Change 0 to 1
                if changes_made < maxChanges:
                    for j in range(n):
                        if j != alice_index and current_nums[j] == 0:
                            next_nums = list(current_nums)
                            next_nums[j] = 1
                            next_state_nums = tuple(next_nums)
                            if next_state_nums not in visited_states:
                                visited_states.add(next_state_nums)
                                queue.append((next_nums, changes_made + 1, ones_picked, moves + 1))
                                
                # Operation 2: Swap adjacent 1 and 0
                for x in range(n - 1):
                    y = x + 1
                    if current_nums[x] == 1 and current_nums[y] == 0:
                        next_nums = list(current_nums)
                        next_nums[x], next_nums[y] = 0, 1
                        picked_increment = 0
                        if y == alice_index:
                            picked_increment = 1
                            next_nums[y] = 0
                        next_state_nums = tuple(next_nums)
                        if next_state_nums not in visited_states:
                            visited_states.add(next_state_nums)
                            queue.append((next_nums, changes_made, ones_picked + picked_increment, moves + 1))
                            
                for y in range(n - 1):
                    x = y + 1
                    if current_nums[y] == 1 and current_nums[x] == 0:
                        next_nums = list(current_nums)
                        next_nums[x], next_nums[y] = 1, 0
                        picked_increment = 0
                        if x == alice_index:
                            picked_increment = 1
                            next_nums[x] = 0
                        next_state_nums = tuple(next_nums)
                        if next_state_nums not in visited_states:
                            visited_states.add(next_state_nums)
                            queue.append((next_nums, changes_made, ones_picked + picked_increment, moves + 1))
                            
            min_total_moves = min(min_total_moves, min_moves_for_index)

        return min_total_moves if min_total_moves != float('inf') else -1

from typing import List