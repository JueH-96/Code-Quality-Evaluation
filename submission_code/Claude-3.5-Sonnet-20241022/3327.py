class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones = []
        # Find positions of all 1s
        for i in range(n):
            if nums[i] == 1:
                ones.append(i)
        
        # If we can create all needed 1s using maxChanges
        if k <= maxChanges:
            # First 1 is free if we start at a 1, otherwise costs 1 move
            if ones:
                return 2 * (k - 1)
            return 2 * k - 1
            
        # We need to use some existing 1s
        # Can use maxChanges new 1s, rest must come from existing
        need_existing = k - maxChanges
        
        if len(ones) < need_existing:
            return -1
            
        min_moves = float('inf')
        # Try each position as starting point
        for i in range(n):
            moves = 0
            existing_used = 0
            changes_left = maxChanges
            
            # First check if we're starting at a 1
            if nums[i] == 1:
                existing_used += 1
                
            # Find closest 1s to position i
            left = right = i
            left_idx = right_idx = 0
            while existing_used < need_existing:
                left_val = right_val = float('inf')
                
                # Find next 1 on left
                while left_idx < len(ones) and ones[left_idx] >= left:
                    left_idx += 1
                if left_idx < len(ones):
                    left_val = abs(ones[left_idx] - i)
                    
                # Find next 1 on right    
                while right_idx < len(ones) and ones[right_idx] <= right:
                    right_idx += 1
                if right_idx < len(ones):
                    right_val = abs(ones[right_idx] - i)
                
                # Take closest 1
                if left_val <= right_val:
                    moves += left_val
                    left = ones[left_idx]
                    existing_used += 1
                else:
                    moves += right_val
                    right = ones[right_idx]
                    existing_used += 1
                    
            # Add moves for remaining changes
            moves += 2 * changes_left
            
            min_moves = min(min_moves, moves)
            
        return min_moves