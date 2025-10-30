import bisect

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        
        # Collect existing ones and potential created ones
        existing = [(i, False) for i in range(n) if nums[i] == 1]
        created = [(i, True) for i in range(n) if nums[i] == 0][:maxChanges]
        
        # Combine and sort positions
        positions = sorted(existing + created, key=lambda x: x[0])
        
        # Compute prefix sums of positions
        prefix = [0] * (len(positions) + 1)
        for i in range(len(positions)):
            prefix[i + 1] = prefix[i] + positions[i][0]
        
        # Compute prefix sums of created positions
        is_created = [1 if pos[1] else 0 for pos in positions]
        prefix_created = [0] * (len(positions) + 1)
        for i in range(len(positions)):
            prefix_created[i + 1] = prefix_created[i] + is_created[i]
        
        min_moves = float('inf')
        
        # Slide the window
        for i in range(len(positions) - k + 1):
            median_index = i + k // 2
            aliceIndex = positions[median_index][0]
            
            # Calculate sum of distances
            sum_distance = (2 * median_index - 2 * i - k) * aliceIndex - 2 * prefix[median_index] + prefix[i] + prefix[i + k]
            
            # Calculate number of created ones in the window
            window_created = prefix_created[i + k] - prefix_created[i]
            
            # Count number of created ones at aliceIndex
            left = bisect.bisect_left(positions, (aliceIndex, False), lo=i, hi=i + k)
            right = bisect.bisect_right(positions, (aliceIndex, True), lo=i, hi=i + k)
            number_of_created_with_distance_zero = prefix_created[right] - prefix_created[left]
            
            # Calculate sum of moves
            sum_moves = sum_distance + 2 * window_created - number_of_created_with_distance_zero
            
            if sum_moves < min_moves:
                min_moves = sum_moves
        
        return min_moves