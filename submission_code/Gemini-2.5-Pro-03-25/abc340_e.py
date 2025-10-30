# YOUR CODE HERE
import sys

# Fenwick Tree (BIT) Implementation using 0-based indexing interface
# Internally uses 1-based indexing for tree structure
# Handles indices from 0 up to max_idx inclusive.
class FenwickTreeZeroBased:
    """
    Fenwick Tree (Binary Indexed Tree) supporting point updates and prefix sum queries.
    Uses 0-based indexing for method calls, but 1-based indexing internally.
    `max_idx` is the maximum index that this tree needs to support operations on.
    The tree can handle indices 0, 1, ..., max_idx. The size needed for the internal
    array is `max_idx + 2`.
    """
    def __init__(self, max_idx):
        # The number of elements indexed 0..max_idx is max_idx + 1. This is the conceptual size.
        self.size = max_idx + 1 
        # Internal tree array needs size `self.size + 1` = `max_idx + 2` to accommodate 1-based indexing up to index `self.size`.
        self.tree = [0] * (self.size + 1) 

    # Add value to index idx (0-based)
    def add(self, idx, value):
        """Adds `value` to the element at index `idx` (0-based).
        This operation effectively updates the prefix sums for indices >= idx."""
        
        # Check bounds defensively (optional but good practice for debugging)
        # if not (0 <= idx < self.size):
        #     raise IndexError(f"Index {idx} out of bounds [0, {self.size - 1}] for Fenwick Tree add operation")

        idx += 1 # Convert to 1-based index for internal tree logic
        # Traverse up the tree structure, updating relevant nodes
        # Loop condition uses <= self.size because the tree array indices are 1..self.size
        while idx <= self.size: 
            self.tree[idx] += value
            # Move to the next index in the BIT structure that includes this index's contribution
            idx += idx & (-idx) 

    # Query prefix sum up to idx (inclusive) (0-based index)
    # returns sum[0...idx]
    def query(self, idx):
        """Computes the prefix sum up to index `idx` (inclusive), i.e., sum(array[0]...array[idx])."""
        
        # Check bounds defensively (optional but good practice)
        # if not (0 <= idx < self.size):
        #     raise IndexError(f"Index {idx} out of bounds [0, {self.size - 1}] for Fenwick Tree query operation")

        idx += 1 # Convert to 1-based index
        s = 0
        # Traverse down the tree structure (towards index 0) summing values from relevant nodes
        while idx > 0:
            s += self.tree[idx]
            # Move to the parent index in the BIT structure
            idx -= idx & (-idx) 
        return s


def solve():
    """
    Solves the ball distribution problem using Fenwick Tree for range updates.
    Maintains base counts in array X, accumulated range updates in BIT,
    and a global offset for balls distributed uniformly across all boxes after full cycles.
    """
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split())) # Initial ball counts
    B = list(map(int, sys.stdin.readline().split())) # Indices of boxes for M operations

    # X stores base values for each box. Initially A_i, these values get modified
    # when a box is emptied to ensure its effective count becomes 0.
    X = list(A) 
    
    # Create a Fenwick Tree. It needs to support indices 0 to N because range updates
    # involving the last box N-1 might require updates at index N (e.g. add(N, -1)).
    # So the maximum index required is N.
    bit = FenwickTreeZeroBased(N) 
    
    # offset stores the cumulative count of balls added to *every* box due to full cycles
    # of distribution (K // N balls per operation).
    offset = 0 

    # Process each of the M operations
    for k in range(M):
        box_idx = B[k] # Get the box index for the k-th operation
        
        # 1. Query the BIT for accumulated range updates affecting box_idx up to this point.
        # The value returned by query(box_idx) represents the net effect of range updates on box_idx.
        current_bit_update = bit.query(box_idx)
        
        # 2. Calculate the effective current ball count K in box_idx.
        # This is the sum of its base value X[box_idx], the BIT updates relevant to it, 
        # and the global offset accumulated so far.
        K = X[box_idx] + current_bit_update + offset
        
        # If the box is empty (K=0), the operation has no effect. Skip to the next operation.
        if K == 0: 
             continue

        # 3. Update the base value X[box_idx].
        # Since all K balls are removed, the effective count of box_idx must become 0.
        # We achieve this by adjusting the base value X[box_idx]. The equation
        # New_X[box_idx] + current_bit_update + offset = 0 must hold.
        # So, we set X[box_idx] = -current_bit_update - offset.
        X[box_idx] = -current_bit_update - offset

        # 4. Update the global offset.
        # The K balls are distributed. K // N balls are distributed in full cycles, meaning
        # each of the N boxes receives K // N balls. We track this via the global offset.
        balls_per_box_full_cycles = K // N
        offset += balls_per_box_full_cycles
        
        # 5. Distribute the remaining R = K % N balls.
        # These R balls are placed one by one into boxes (box_idx + 1)%N, (box_idx + 2)%N, ..., (box_idx + R)%N.
        # This is equivalent to adding 1 to each box in this range (modulo N).
        # We implement this using range updates on the BIT.
        R = K % N
        
        if R > 0:
            # Calculate the start and end indices of the range receiving 1 ball each.
            start_idx = (box_idx + 1) % N
            end_idx = (box_idx + R) % N
            
            # Apply range update using BIT difference array technique: add 1 to the specified range.
            # This involves two point updates on the BIT.
            if start_idx <= end_idx:
                # Case 1: The range [start_idx, end_idx] does not wrap around N.
                bit.add(start_idx, 1)  # Increment at the start of the range
                # Decrement at end_idx + 1 to cancel the effect for indices > end_idx.
                # The index end_idx + 1 can be up to N. Our BIT supports index N.
                bit.add(end_idx + 1, -1) 
            else:
                # Case 2: The range wraps around N. This requires two logical range updates:
                # [start_idx, N-1] and [0, end_idx].
                
                # Update for range [start_idx, N-1]
                bit.add(start_idx, 1) # Increment at start_idx
                # Decrement at index N to cancel the effect for indices >= N. Index N is valid.
                bit.add(N, -1) 
                
                # Update for range [0, end_idx]
                bit.add(0, 1) # Increment at index 0
                # Decrement at index end_idx + 1 to cancel the effect for indices > end_idx.
                bit.add(end_idx + 1, -1)


    # After processing all M operations, calculate the final ball counts for each box.
    final_counts = []
    for j in range(N):
        # The final count for box j is the sum of its final base value X[j],
        # the total accumulated BIT update affecting box j (obtained by query(j)),
        # and the final global offset.
        final_bit_update = bit.query(j) 
        final_count = X[j] + final_bit_update + offset 
        final_counts.append(final_count)

    # Print the final counts separated by spaces.
    print(*(final_counts))

# Execute the solve function to run the main logic of the program.
solve()