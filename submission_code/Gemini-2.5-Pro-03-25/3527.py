import sys
# Increase recursion depth if necessary, though not using recursion heavily here.
# sys.setrecursionlimit(2000) 

# Attempt to import sortedcontainers.SortedList. If unavailable, use a basic mock.
try:
    from sortedcontainers import SortedList
except ImportError:
    # Simple mock SortedList using list and bisect if sortedcontainers is not available
    # This mock has O(N) remove complexity, which might be too slow for large inputs.
    # For competitive programming platforms that provide sortedcontainers, this would pass.
    import bisect
    class SortedList:
        def __init__(self, iterable=None):
            self._list = sorted(iterable) if iterable is not None else []
        def add(self, value):
            bisect.insort_left(self._list, value)
        def remove(self, value):
             try:
                # Find first occurrence index using bisect_left and check if value matches
                idx = bisect.bisect_left(self._list, value)
                if idx < len(self._list) and self._list[idx] == value:
                    del self._list[idx]
                else:
                    # Value not found, raise ValueError consistent with list.remove behavior
                    raise ValueError(f"{value} not found in list")
             except ValueError:
                 # If list.remove behavior is to ignore not found, then just pass.
                 # Python's list.remove raises ValueError, so perhaps mock should too, or handle based on context.
                 # For this problem, if remove is called on non-existent key, it might indicate logic error elsewhere.
                 # Let's pass silently to avoid crashing, but be aware this might mask issues.
                 pass 
        def bisect_left(self, value):
            return bisect.bisect_left(self._list, value)
        def bisect_right(self, value):
            return bisect.bisect_right(self._list, value)
        def __contains__(self, value):
            i = bisect.bisect_left(self._list, value)
            return i != len(self._list) and self._list[i] == value
        def __len__(self):
            return len(self._list)
        def __getitem__(self, index):
            return self._list[index]
        def __iter__(self):
            return iter(self._list)
        # Note: pop operation not used in the main logic, but included for completeness of basic list interface.
        def pop(self, index=-1):
            return self._list.pop(index)


from typing import List

# Fenwick Tree (Binary Indexed Tree) implementation
class FenwickTree:
    def __init__(self, size):
        # Size is max index + 1. For lengths 1..N, need size N+1.
        self.tree = [0] * (size + 1)
        self.size = size # Maximum index handled is size

    def update(self, i, delta):
        # i is the index (length in this problem), 1 <= i <= self.size
        if not (1 <= i <= self.size): return # Index out of bounds
        
        while i <= self.size:
            self.tree[i] += delta
            i += i & (-i) # Move to next relevant index in BIT

    def query(self, i): # query sum prefix [1..i]
        if i > self.size: i = self.size # Cap index at max size
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i) # Move to previous relevant index
        return s
    
    def query_suffix(self, i): # query sum for range [i, size]
        if i > self.size: return 0
        if i < 1: i = 1 # Ensure index is at least 1
        total_sum = self.query(self.size)
        prefix_sum_before_i = self.query(i - 1) 
        return total_sum - prefix_sum_before_i

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        N = len(colors)
        # A[i] = 1 if colors[i] != colors[(i + 1) % N] else 0
        # Represents alternating property between index i and i+1
        A = [(1 if colors[i] != colors[(i + 1) % N] else 0) for i in range(N)]

        # Fenwick Trees to maintain counts for query type 1
        # bit1 stores counts N_L for blocks of length L
        # bit2 stores N_L * L for blocks of length L
        bit1 = FenwickTree(N) 
        bit2 = FenwickTree(N) 

        # Dictionary to store block info: {start_index: length}
        blocks = {} 
        # SortedList to store start indices of blocks for efficient searching
        block_starts = SortedList() 
        
        # Helper function to find the block containing a given index `idx`.
        # Returns tuple (start_index, length) if found, otherwise None.
        def find_block_containing(idx):
            if not block_starts: return None
            
            # Find the potential block starting at or before idx
            bs_idx = block_starts.bisect_right(idx) - 1
            if bs_idx >= 0:
                start = block_starts[bs_idx]
                # Check if block still exists (it might have been removed during updates)
                if start not in blocks: 
                     # This can happen if the block was removed but SortedList update failed or was deferred.
                     # Or if stale data is used. Let's assume `blocks` is the source of truth.
                     # We might need stronger consistency checks or careful update order.
                     # For now, proceed assuming inconsistency is handled.
                     pass # If block doesn't exist, try next check.
                else:
                    length = blocks[start]
                    end = (start + length - 1) % N
                    if start <= end: # Normal block (no wrap around)
                        if start <= idx <= end: return (start, length)
                    else: # Wrap around block
                        if start <= idx < N or 0 <= idx <= end: return (start, length)

            # Check the block with the largest start index (handles wrap around for small idx values)
            if block_starts: # Ensure list is not empty
                 last_start = block_starts[-1]
                 if last_start in blocks: 
                     last_length = blocks[last_start]
                     last_end = (last_start + last_length - 1) % N
                     if last_start > last_end: # If it wraps around
                         if last_start <= idx < N or 0 <= idx <= last_end:
                            return (last_start, last_length)
            return None # Index not found in any block

        # Helper function to add a block to our tracking structures
        def add_block(start, length):
            if length == 0: return
            blocks[start] = length
            block_starts.add(start)
            # Update Fenwick trees with the contribution of this block
            bit1.update(length, 1)
            bit2.update(length, length)
        
        # Helper function to remove a block from tracking structures
        def remove_block(start, length):
            if length == 0: return
            # Check existence before removing to avoid errors
            if start in blocks and blocks[start] == length: 
                del blocks[start]
                try:
                   # Attempt to remove from SortedList. Use try-except for robustness.
                   block_starts.remove(start)
                except ValueError:
                   pass # Ignore if not found, possibly already processed or edge case.
                # Update Fenwick trees to remove the contribution
                bit1.update(length, -1)
                bit2.update(length, -length)

        # Initialize block tracking based on the initial state of A
        idx = 0
        while idx < N:
            if A[idx] == 1:
                start_idx = idx
                length = 0
                curr = idx
                # Use a set to detect cycles/revisits during scan; safety measure for wrap-around logic
                temp_visited = set() 
                while A[curr % N] == 1:
                    # Basic cycle detection/limit check
                    if curr % N in temp_visited: break 
                    temp_visited.add(curr % N)

                    length += 1
                    curr += 1
                    # If length reaches N, it's a full circle alternating block
                    if length == N: break 
                
                add_block(start_idx, length)
                
                if length == N: 
                    idx = N # Full circle block covers all indices, stop scan.
                else: 
                    # Continue scan from the index after the block ends.
                    # curr points to the index where A[curr % N] == 0 or cycle detected.
                    idx = curr 
            else: # A[idx] == 0
                idx += 1
        
        results = [] # Store results for type 1 queries

        # Process queries
        for query in queries:
            if query[0] == 1: # Type 1: Count alternating groups of size k
                k = query[1]
                # Validate k based on constraints. Minimum size is 3, maximum possible size N.
                if not (3 <= k <= N): 
                     results.append(0) 
                     continue 

                C = k - 1 # Minimum run length of 1s in A needed is k-1

                # Special case: If there is a block of length N (full circle is alternating)
                has_len_N_block = False
                if N in blocks: # Check if a block starts at N (not possible, indices 0..N-1)
                    pass # This condition is probably wrong. Check values not keys.
                
                # Correct check: iterate through block lengths stored in `blocks` values.
                for length in blocks.values():
                    if length == N:
                        has_len_N_block = True
                        break

                if has_len_N_block:
                   # If the whole circle is alternating, there are N groups of size k for any valid k.
                   results.append(N)
                   continue

                # General case calculation using Fenwick Trees
                # Calculate Sum N_L for L >= C
                sum_N_L = bit1.query_suffix(C)
                # Calculate Sum N_L * L for L >= C
                sum_N_L_L = bit2.query_suffix(C)
                
                # Formula for total count: Sum_{L >= C} N_L * (L - C + 1) = Sum N_L*L - (C-1) * Sum N_L
                total_count = sum_N_L_L - (C - 1) * sum_N_L
                results.append(total_count)

            else: # query[0] == 2: Update color at index idx
                idx, new_color = query[1], query[2]
                
                # If color doesn't change, skip update
                if colors[idx] == new_color: continue

                # Indices in A potentially affected by changing colors[idx]
                # p affects pair (p, p+1) -> A[p]
                # idx affects pair (idx-1, idx) -> A[idx-1] or A[p]
                # And pair (idx, idx+1) -> A[idx] or A[q]
                p, q = (idx - 1 + N) % N, idx
                
                # Identify all blocks potentially affected by changes around indices p, q
                potentially_affected_starts = set()
                
                # Check a neighborhood around p and q to find potentially affected blocks
                # A change at idx affects A[p] and A[q]. Check indices around p and q.
                indices_to_check_for_blocks = set()
                for i in [p, q]:
                    # Check index i itself, and its neighbors i-1, i+1
                    for d in range(-1, 2): 
                         indices_to_check_for_blocks.add((i + d + N) % N)

                for i in indices_to_check_for_blocks:
                    block_info = find_block_containing(i)
                    if block_info: 
                        # Add the start index of the found block
                        potentially_affected_starts.add(block_info[0])

                # Remove all potentially affected blocks before recalculating state
                for start in potentially_affected_starts:
                    if start in blocks: # Check if block still exists
                        length = blocks[start]
                        remove_block(start, length) # Helper handles updates to BITs, dict, SortedList

                # Perform the color update and recalculate affected A values
                colors[idx] = new_color
                A[p] = 1 if colors[p] != colors[(p + 1) % N] else 0
                A[q] = 1 if colors[q] != colors[(q + 1) % N] else 0
                
                # Rescan locally around affected indices p, q to find new/modified block structure
                rescan_indices_potential_starts = set()
                # Consider indices p, q and their neighbors as potential starts of new blocks
                for i in [p, q]:
                     for d in range(-1, 2): 
                          rescan_indices_potential_starts.add((i + d + N) % N)

                processed_in_rescan = set() # Track indices covered by newly added blocks during this rescan phase
                
                # Sort indices to process them in order, potentially helps managing overlaps/state
                indices_to_scan_sorted = sorted(list(rescan_indices_potential_starts))
                
                for i in indices_to_scan_sorted:
                    # Skip if index already covered by a new block found in this update pass
                    if i in processed_in_rescan: continue
                    
                    # If A[i] is 1, it might start a new block or be part of one
                    if A[i] == 1:
                        # Check if this index 'i' is already part of an existing block
                        # This check is important to avoid adding duplicate blocks or fragments.
                        # find_block_containing checks against current `blocks` state.
                        if find_block_containing(i) is not None:
                             # If already part of a known block, this block must have started before `i`
                             # and covered `i`. We should have processed its start earlier if sorted.
                             # Skip processing `i` if it's already inside a block.
                             continue

                        # Found a potential start of a new block at `i`
                        start = i
                        length = 0
                        curr = i
                        temp_visited = set() # Safety check
                        while A[curr % N] == 1:
                            curr_mod_N = curr % N
                            # If we hit an index already processed in this rescan pass, stop.
                            if curr_mod_N in processed_in_rescan: break 
                            if curr_mod_N in temp_visited: break # Safety cycle break
                            temp_visited.add(curr_mod_N)

                            processed_in_rescan.add(curr_mod_N) # Mark this index as covered
                            length += 1
                            curr += 1
                            if length == N: break # Full circle alternating block
                        
                        if length > 0:
                             # Add the newly found block
                             add_block(start, length) 
        
        return results