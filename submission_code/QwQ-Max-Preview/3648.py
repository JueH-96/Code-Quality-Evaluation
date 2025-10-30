from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        if n == 0:
            return 0
        
        # Initialize DP table. The state is (i1, j1, j2, i3) for step k.
        # We'll use a dictionary to track the maximum sum for each state at step k.
        dp = {}
        initial_i1, initial_j1 = 0, 0
        initial_j2 = n - 1
        initial_i3 = n - 1
        initial_j3 = 0
        
        # Step 0: initial positions
        # Check which cells are visited in the initial step
        visited = set()
        sum_initial = 0
        if (initial_i1, initial_j1) not in visited:
            sum_initial += fruits[initial_i1][initial_j1]
            visited.add((initial_i1, initial_j1))
        if (0, initial_j2) not in visited:
            sum_initial += fruits[0][initial_j2]
            visited.add((0, initial_j2))
        if (initial_i3, 0) not in visited:
            sum_initial += fruits[initial_i3][0]
            visited.add((initial_i3, 0))
        dp_key = (initial_i1, initial_j1, initial_j2, initial_i3)
        dp[dp_key] = (sum_initial, visited)
        
        for step in range(1, n):
            new_dp = {}
            for state, (current_sum, visited) in dp.items():
                i1, j1, j2, i3 = state
                
                # Generate possible moves for child1
                child1_moves = []
                if i1 + 1 < n and j1 + 1 < n:
                    child1_moves.append((i1 + 1, j1 + 1))
                if i1 + 1 < n:
                    child1_moves.append((i1 + 1, j1))
                if j1 + 1 < n:
                    child1_moves.append((i1, j1 + 1))
                
                # Generate possible moves for child2
                child2_moves = []
                new_i2 = step  # child2's i is step
                new_j2_options = []
                for dj in [-1, 0, 1]:
                    new_j2 = j2 + dj
                    if 0 <= new_j2 < n:
                        new_j2_options.append(new_j2)
                child2_moves = [(new_i2, new_j2) for new_j2 in new_j2_options]
                
                # Generate possible moves for child3
                child3_moves = []
                new_j3 = step  # child3's j is step
                new_i3_options = []
                for di in [-1, 0, 1]:
                    new_i3 = i3 + di
                    if 0 <= new_i3 < n:
                        new_i3_options.append(new_i3)
                child3_moves = [(new_i3, new_j3) for new_i3 in new_i3_options]
                
                # Iterate over all possible moves for child1, child2, child3
                for new_i1, new_j1 in child1_moves:
                    for new_i2, new_j2 in child2_moves:
                        for new_i3, new_j3 in child3_moves:
                            # Check if new positions are valid
                            if new_i1 >= 0 and new_j1 >= 0 and new_i2 >= 0 and new_j2 >= 0 and new_i3 >= 0 and new_j3 >= 0:
                                # Compute the new sum
                                new_visited = set(visited)
                                sum_add = 0
                                # Check each new position
                                if (new_i1, new_j1) not in new_visited:
                                    sum_add += fruits[new_i1][new_j1]
                                    new_visited.add((new_i1, new_j1))
                                if (new_i2, new_j2) not in new_visited:
                                    sum_add += fruits[new_i2][new_j2]
                                    new_visited.add((new_i2, new_j2))
                                if (new_i3, new_j3) not in new_visited:
                                    sum_add += fruits[new_i3][new_j3]
                                    new_visited.add((new_i3, new_j3))
                                new_sum = current_sum + sum_add
                                new_state = (new_i1, new_j1, new_j2, new_i3)
                                if new_state not in new_dp or new_sum > new_dp[new_state][0]:
                                    new_dp[new_state] = (new_sum, new_visited.copy())
            dp = new_dp
        
        max_sum = 0
        for state, (current_sum, _) in dp.items():
            if state[0] == n-1 and state[1] == n-1 and state[2] == n-1 and state[3] == n-1:
                if current_sum > max_sum:
                    max_sum = current_sum
        return max_sum