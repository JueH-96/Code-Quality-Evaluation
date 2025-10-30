class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        totalOnesAvailable = sum(nums) + maxChanges
        if totalOnesAvailable < k:
            return -1  # Impossible to collect k ones
        aliceIndex = 0  # We can choose any index [0, n-1]
        # To minimize initial cost, we can try all possible aliceIndex
        # But due to constraints, choosing the median of ones may be better
        # Since constraints are tight, we need to fix aliceIndex
        
        # For efficiency, let's try to fix aliceIndex at position of a one
        # or simply at the middle
        # Alternatively, we can try to minimize the total cost by choosing aliceIndex
        # that minimizes sum of distances

        # Let's try to choose aliceIndex as the median of the positions we will collect
        # First, get all positions of nums

        positions_of_ones = [i for i, val in enumerate(nums) if val == 1]
        positions_of_zeros = [i for i, val in enumerate(nums) if val == 0]
        
        total_positions = len(nums)
        # Let's try aliceIndex at different positions to minimize total cost
        # Since further optimization may be too heavy, let's fix aliceIndex at 0
        aliceIndex = 0
        min_total_cost = float('inf')
        
        # Alternatively, we can pick aliceIndex at positions_of_ones[k//2] if possible
        # Since we cannot try all positions, let's limit to a few positions
        possible_alice_indices = [0, n // 2, n - 1]
        if positions_of_ones:
            possible_alice_indices.append(positions_of_ones[len(positions_of_ones)//2])
        
        for aliceIndex in possible_alice_indices:
            initial_one = int(nums[aliceIndex] == 1)
            k_remaining = k - initial_one
            if k_remaining < 0:  # We have already picked up enough ones
                return 0
            positions = []
            for i in range(n):
                if i == aliceIndex:
                    continue  # Skip aliceIndex
                distance = abs(i - aliceIndex)
                if nums[i] == 1:
                    cost_p = distance
                    positions.append((cost_p, False))  # False indicates no need for action type 1
                else:
                    cost_p = 1 + distance  # 1 for action type 1
                    positions.append((cost_p, True))  # True indicates need for action type 1
            # Sort positions by cost_p
            positions.sort()
            total_cost = 0
            action_type1_used = 0
            positions_selected = 0
            for cost_p, need_change in positions:
                if positions_selected == k_remaining:
                    break
                if need_change:
                    if action_type1_used < maxChanges:
                        action_type1_used += 1
                        positions_selected +=1
                        total_cost += cost_p
                    else:
                        continue  # Cannot select positions needing action type 1
                else:
                    positions_selected +=1
                    total_cost += cost_p
            if positions_selected == k_remaining:
                # Add initial cost if any
                min_total_cost = min(min_total_cost, total_cost)
        return min_total_cost