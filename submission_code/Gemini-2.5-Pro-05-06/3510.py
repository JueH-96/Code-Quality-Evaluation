import sys

class Solution:
    def maximumTotalSum(self, maximumHeight: list[int]) -> int:
        # Sort maximumHeight in descending order. This allows us to greedily
        # assign heights. Towers with higher maximum capacity are considered first.
        maximumHeight.sort(reverse=True)
        
        # used_map[x] = y means that if height x is queried for availability,
        # the actual (potentially smaller) height to check is y. This is because
        # x itself (and heights between y and x) are already taken.
        # This map implements a Disjoint Set Union (DSU) like structure
        # where each set represents a contiguous block of used heights.
        # The value y is the largest height in the block {y, y+1, ..., x-1} if x points to y, or y is available.
        used_map = {} 

        # Iterative find_vacant function to find the largest available positive height <= val.
        # It uses path compression for efficiency.
        def find_vacant_iterative(val: int) -> int:
            # If val is not positive, no positive height can be assigned.
            if val <= 0:
                return 0
            
            path_to_compress = []
            current_val = val
            
            # Trace the chain of used heights.
            # If current_val is in used_map, it means current_val is used.
            # The value used_map[current_val] is the next height to try.
            while current_val in used_map:
                path_to_compress.append(current_val)
                current_val = used_map[current_val]
                # Optimization: if the chain leads to a non-positive number,
                # it means all positive numbers on this path are used up.
                if current_val <= 0:
                    break 
            
            # current_val is now the largest available height <= original val
            # (or current_val <= 0 if no positive height is available).
            
            # Path compression: make all nodes on the traced path point to current_val.
            # This speeds up future queries for these nodes.
            for node in path_to_compress:
                used_map[node] = current_val
            
            return current_val

        total_sum = 0
        
        for max_h_for_current_tower in maximumHeight:
            # Find the largest positive height <= max_h_for_current_tower that is not used.
            assignable_h = find_vacant_iterative(max_h_for_current_tower)
            
            if assignable_h <= 0: # Check includes assignable_h == 0
                # Cannot assign a positive height for this tower that satisfies constraints.
                # This implies it's impossible to assign heights to all towers.
                return -1
            
            total_sum += assignable_h
            
            # Mark assignable_h as used.
            # The DSU logic here means: if assignable_h is now used, any future query for assignable_h
            # should instead check for assignable_h - 1 (or whatever height that resolves to).
            # So, we link assignable_h to assignable_h - 1.
            # The find_vacant_iterative function will correctly resolve this chain if assignable_h - 1
            # is also used or leads to another chain of used heights.
            used_map[assignable_h] = assignable_h - 1

        return total_sum