# YOUR CODE HERE
import sys

# Fenwick tree (Binary Indexed Tree) implementation
class FenwickTree:
    """Fenwick tree class supporting point updates and prefix sum queries."""
    def __init__(self, size):
        """Initializes Fenwick tree for elements indexed 0 to size-1.
        Tree array size is size+1 for 1-based indexing logic.
        """
        self.tree = [0] * (size + 1)
        self.size = size # Maximum index is size-1

    def update(self, i, delta):
        """Adds delta to the element at index i (0-based)."""
        if not (0 <= i < self.size):
            # Optional: handle invalid index, e.g., raise error or log warning.
            # For this problem, indices should always be valid by logic.
            return 

        i += 1 # Convert to 1-based index for BIT internal operations
        while i <= self.size:
            self.tree[i] += delta
            # Move to the next index that includes the effect of index i
            i += i & (-i) 

    def query(self, i):
        """Queries the prefix sum up to index i (0-based, inclusive)."""
        if i < 0:
            return 0
        
        # Ensure query index does not exceed maximum valid index (size-1)
        i = min(i, self.size - 1) 
        
        i += 1 # Convert to 1-based index for BIT internal operations
        s = 0
        while i > 0:
            s += self.tree[i]
            # Move to the index responsible for the next range part in BIT structure
            i -= i & (-i) 
        return s

# Function to calculate the initial total inversion number of a permutation
def calculate_initial_inversions(P, N):
    """Calculates the inversion number of permutation P of length N using Fenwick Tree."""
    # The Fenwick Tree operates on values. Permutation values are 1 to N.
    # We need size N+1 to handle value N, mapping value N to index N in the BIT.
    ft = FenwickTree(N + 1)
    inversions = 0
    # Iterate through the permutation elements
    for i in range(N):
        val = P[i] # Current element value (1-based)
        
        # Count elements seen *before* index i with value greater than P[i]
        # Total elements processed so far = i
        # Elements processed with value <= P[i] = ft.query(val) 
        # The query ft.query(val) sums counts for values from 1 up to val.
        # Number of elements seen > P[i] = Total seen - Elements seen <= P[i]
        inversions += i - ft.query(val) 
        
        # Mark the current value P[i] as seen by incrementing its count in the BIT
        ft.update(val, 1) 
    return inversions

def solve():
    """Reads input, performs the specified operations, and prints inversion counts after each step."""
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    M = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the initial total inversion number of P
    current_inversions = calculate_initial_inversions(P, N)

    # Initialize Fenwick tree for tracking adjacent inversions.
    # An adjacent inversion exists at index k if P[k] > P[k+1].
    # Indices for comparison are k = 0 to N-2. There are N-1 such pairs.
    # The Fenwick Tree size needs to be N-1.
    ft_adj = FenwickTree(N - 1) 
    
    # Keep track of the status (0 or 1) of each adjacent pair inversion P[k] > P[k+1]
    # This helps avoid querying the Fenwick Tree for current status.
    adj_inv_status = [0] * (N - 1) 
    for k in range(N - 1):
        if P[k] > P[k+1]:
            adj_inv_status[k] = 1
            ft_adj.update(k, 1) # Update ft_adj at index k (0-based)

    results = [] # Store inversion counts after each operation

    # Process the M operations specified by sequence A
    for k_idx in range(M):
        K = A[k_idx] # K is from the problem statement (1-based length, 2 <= K <= N)
        
        # Calculate the number of swaps S_K for Operation K.
        # This equals the count of adjacent inversions (P[j] > P[j+1]) for j = 0..K-2.
        # Query the Fenwick tree for the sum over range [0, K-2].
        num_swaps = ft_adj.query(K - 2) 

        # Each swap decreases the total inversion count by 1.
        current_inversions -= num_swaps
        results.append(current_inversions)

        # Simulate Operation K on the permutation P and update the adjacent inversion Fenwick tree (ft_adj).
        # Operation K iterates i = 1..K-1 (problem statement).
        # This corresponds to 0-based array indices j = 0..K-2.
        
        for j in range(K - 1): # Iterate through relevant pairs (P[j], P[j+1])
            if P[j] > P[j+1]:
                # An adjacent inversion exists, so a swap will occur.
                
                # Store original values before the swap, needed for status checks
                Pj_orig = P[j]
                Pjp1_orig = P[j+1]

                # Update adjacent inversion status array and Fenwick tree based on the effects of this swap.

                # -- Update for index j --
                # The pair (j, j+1) was inverted (P[j] > P[j+1]). After swap, P[j+1] < P[j].
                # The inversion at index j disappears. Status changes from 1 to 0.
                # Update ft_adj by -1 at index j. Update status array.
                # Need to check adj_inv_status[j] == 1 first, though it should always be true here.
                if adj_inv_status[j] == 1: 
                    ft_adj.update(j, -1)
                    adj_inv_status[j] = 0

                # -- Update for index j-1 --
                # Check if the swap affects the adjacent pair (j-1, j). This happens if j > 0.
                if j > 0:
                    # Old status depended on comparison: P[j-1] > P[j]_orig
                    old_status_jm1 = adj_inv_status[j-1]
                    # New status depends on comparison: P[j-1] > P[j+1]_orig (since P[j] takes value P[j+1]_orig after swap)
                    new_status_jm1 = 1 if P[j-1] > Pjp1_orig else 0
                    # If status changed, update ft_adj and status array
                    if new_status_jm1 != old_status_jm1:
                        delta = new_status_jm1 - old_status_jm1
                        ft_adj.update(j-1, delta)
                        adj_inv_status[j-1] = new_status_jm1
                
                # -- Update for index j+1 --
                # Check if the swap affects the adjacent pair (j+1, j+2). This happens if j+1 < N-1.
                if j + 1 < N - 1: # Check index j+1 is valid for pair (j+1, j+2)
                     # Old status depended on comparison: P[j+1]_orig > P[j+2]
                     old_status_jp1 = adj_inv_status[j+1]
                     # New status depends on comparison: P[j]_orig > P[j+2] (since P[j+1] takes value P[j]_orig after swap)
                     # P[j+2] value is unchanged by the swap at j.
                     Pjp2 = P[j+2] 
                     new_status_jp1 = 1 if Pj_orig > Pjp2 else 0 
                     # If status changed, update ft_adj and status array
                     if new_status_jp1 != old_status_jp1:
                         delta = new_status_jp1 - old_status_jp1
                         ft_adj.update(j+1, delta)
                         adj_inv_status[j+1] = new_status_jp1

                # Perform the actual swap in the permutation array P
                P[j], P[j+1] = Pjp1_orig, Pj_orig

    # Print all computed inversion counts, one per line
    for res in results:
        print(res)

# Execute the main solver function
solve()