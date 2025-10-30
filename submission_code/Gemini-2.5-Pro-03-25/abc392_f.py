# YOUR CODE HERE
import sys

# Fenwick tree (BIT) implementation
# Uses 1-based indexing internally
class FenwickTree:
    """ Fenwick Tree (Binary Indexed Tree) supporting point updates and prefix queries. """
    def __init__(self, size):
        """ Initializes a Fenwick tree of given size. All elements initially 0. """
        # Initialize tree with zeros. Size is the max index supported.
        # The tree array has size + 1 elements to accommodate 1-based indexing.
        self.tree = [0] * (size + 1)
        self.size = size

    def update(self, idx, val):
        """ Adds value val to element at index idx. Time complexity: O(log N). """
        # Ensure index is within valid range [1, size]
        if not (1 <= idx <= self.size):
             raise ValueError("Index must be between 1 and size for update.")
        
        while idx <= self.size:
            self.tree[idx] += val
            # Move to the next index that idx contributes to in the BIT structure
            # This is done by adding the least significant bit (LSB) of idx
            idx += idx & (-idx) 

    def query(self, idx):
        """ Computes the prefix sum up to index idx (sum of elements from 1 to idx). Time complexity: O(log N). """
        # Ensure index is within valid range [0, size]. query(0) should return 0.
        if not (0 <= idx <= self.size):
             raise ValueError("Index must be between 0 and size for query.")
             
        s = 0
        while idx > 0:
            s += self.tree[idx]
            # Move to the parent index in the implicit tree structure
            # This is done by subtracting the LSB of idx
            idx -= idx & (-idx) 
        return s

    def find_kth(self, k):
        """ Finds the smallest index `idx` such that query(idx) >= k. Time complexity: O(log N).
        Assumes the values stored (conceptually) in the array are non-negative.
        Requires that such an index exists (i.e., k <= total sum currently in BIT).
        k must be positive.
        """
        # Basic validation for k
        if k <= 0:
             raise ValueError("k must be positive for find_kth.")

        idx = 0
        
        # Determine the largest power of 2 less than or equal to self.size
        # This mask will be used to traverse the implicit tree structure like binary search
        mask = 1
        # Find the highest power of 2 <= self.size efficiently
        while (mask << 1) <= self.size:
             mask <<= 1
        
        # Traverse down from the conceptual root of the BIT structure
        while mask > 0:
            # Calculate potential next index by adding the current power of 2 `mask`
            t_idx = idx + mask 
            
            # Check if this potential index is within the bounds of the tree
            if t_idx <= self.size:
                 # self.tree[t_idx] stores the sum of a specific range ending at t_idx.
                 # If the sum contributed by this range (`self.tree[t_idx]`) is less than the remaining `k`,
                 # it means the k-th element we are looking for must be in an index greater than `t_idx`.
                 if self.tree[t_idx] < k:
                      # Subtract the sum of this range from k, as we are effectively skipping this range.
                      k -= self.tree[t_idx]
                      # Move the current index `idx` to `t_idx`, effectively moving our search window to the right.
                      idx = t_idx
            
            # Move to the next smaller power of 2 for the next step of binary search
            mask >>= 1 

        # After the loop, `idx` holds the largest index such that `query(idx)` is strictly less than the original `k`.
        # Therefore, the smallest index `res` such that `query(res) >= k` is `idx + 1`.
        return idx + 1


def solve():
    """ Reads input, solves the problem using Fenwick tree, and prints the result. """
    N = int(sys.stdin.readline())
    # Read P list. P contains 1-based positions P_1, ..., P_N
    P = list(map(int, sys.stdin.readline().split()))

    # Initialize Fenwick Tree of size N.
    ft = FenwickTree(N)
    
    # Initialize all N slots as available (value 1).
    # This step effectively builds the BIT representing an array of N ones.
    # It takes O(N log N) time.
    for i in range(1, N + 1):
        ft.update(i, 1)

    # Result array to store the final configuration of array A
    Result = [0] * N

    # Process indices in reverse order, from N down to 1
    # This simulates the insertion process in reverse time, which allows us
    # to determine the final position of each element i using P_i.
    for i in range(N, 0, -1):
        # Get the target rank P_i. Note that P is 0-indexed list from input,
        # so P_i corresponds to P[i-1].
        target_rank = P[i-1] 
        
        # Find the 1-based index k of the target_rank-th available slot using find_kth.
        # Available slots are those with value 1 in the conceptual array represented by the BIT.
        k = ft.find_kth(target_rank)
        
        # Place element i into the found slot k. Use k-1 for 0-based Result array index.
        Result[k-1] = i
        
        # Mark the k-th slot as occupied by updating its value in the BIT to 0.
        # This is achieved by adding -1 to the value at index k.
        ft.update(k, -1)

    # Print the final array elements separated by spaces.
    # The * operator unpacks the list elements as individual arguments to print.
    print(*(Result))


# Execute the solve function when the script is run
if __name__ == '__main__':
    solve()