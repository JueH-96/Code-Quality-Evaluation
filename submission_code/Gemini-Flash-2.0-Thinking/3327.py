class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones_indices = [i for i, num in enumerate(nums) if num == 1]
        num_ones = len(ones_indices)
        min_moves = float('inf')

        for aliceIndex in range(n):
            for num_initial_ones in range(max(0, k - maxChanges), min(k, num_ones) + 1):
                num_changes = k - num_initial_ones
                if 0 <= num_changes <= maxChanges:
                    cost = num_changes

                    if num_initial_ones == 0 and k > 0:
                        min_moves = min(min_moves, cost)
                        continue

                    if num_initial_ones > 0:
                        from itertools import combinations
                        for initial_ones_picked_indices in combinations(ones_indices, num_initial_ones):
                            movement_cost = 0
                            for one_index in initial_ones_picked_indices:
                                movement_cost += abs(one_index - aliceIndex)
                            min_moves = min(min_moves, cost + movement_cost)
                    elif k == 0:
                        min_moves = min(min_moves, 0)

        if min_moves == float('inf'):
            return -1 # Should not happen based on constraints
        return min_moves