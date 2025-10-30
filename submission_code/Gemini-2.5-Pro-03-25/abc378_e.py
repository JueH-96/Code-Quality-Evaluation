# YOUR CODE HERE
import sys

# Fenwick Tree class (Binary Indexed Tree)
class FenwickTree:
    """
    A Fenwick Tree (Binary Indexed Tree) supports point updates and prefix sum queries
    in O(log N) time. This implementation uses 1-based indexing internally.
    It is used here to count inversions efficiently.
    """
    def __init__(self, size):
        """
        Initializes a Fenwick Tree for indices 1 to size.
        The internal array `tree` stores the BIT structure and has size+1 elements
        to accommodate 1-based indexing up to `size`.
        :param size: The maximum index value the tree will support. In this problem, it's M.
        """
        # The tree array needs size+1 elements for 1-based indexing. Indices are 1..size.
        self.tree = [0] * (size + 1)
        # Store the maximum index this tree supports (which is 'size')
        self.max_idx = size 

    def add(self, idx, val):
        """
        Adds 'val' to the element count at index 'idx'. Represents adding an element with value idx-1.
        :param idx: The 1-based index to update. Should be between 1 and self.max_idx.
        :param val: The value to add (typically 1 for frequency counting).
        """
        # Check bounds for safety, although logic should guarantee validity.
        # If idx is out of bounds [1, M], something is wrong.
        if not (1 <= idx <= self.max_idx):
            # This case should not happen with correct logic `value % M + 1`.
            # Could raise an error for debugging if needed.
             return 

        while idx <= self.max_idx: # Iterate through indices to update in the BIT structure
            self.tree[idx] += val
            # Move to the next index that covers a larger range including idx
            # `idx & (-idx)` isolates the value of the least significant bit set in idx
            idx += idx & (-idx) 

    def query(self, idx):
        """
        Queries the cumulative frequency up to index 'idx' (sum of counts for indices 1 to idx).
        This corresponds to the count of elements with values <= idx-1.
        :param idx: The 1-based index up to which the sum is required.
        :return: The cumulative sum of frequencies from index 1 to idx.
        """
        # Handle edge cases for query index. Querying below index 1 should yield 0.
        if idx < 1:
            return 0
        # Clamp query index if it exceeds the maximum supported index M.
        idx = min(idx, self.max_idx)

        s = 0
        while idx > 0: # Iterate down through relevant tree nodes summing counts
            s += self.tree[idx]
            # Move to the index representing the range ending just before the current LSB range start
            idx -= idx & (-idx) 
        return s

def solve():
    """
    Solves the main problem using the derived formula:
    Total Sum = S_total - M * Q_sum + M * Inv
    
    Where:
    - Total Sum is the target value: Sum_{1<=l<=r<=N} ( (Sum_{i=l..r} A_i) mod M ).
    - S_total is the sum of all subarray sums: Sum_{1<=l<=r<=N} Sum_{i=l..r} A_i.
    - Q_sum is derived from the floor division part: Sum_{r=1..N, j=0..r-1} floor( (P_r - P_j) / M ).
      It simplifies to N*q[N] + Sum_{r=1..N-1} (2*r - N) * q[r].
    - Inv is the number of pairs (j, r) with 0 <= j < r <= N such that P_r % M < P_j % M.
      Calculated using a Fenwick tree.
    - P_k is the prefix sum Sum_{i=1..k} A_i.
    - q_k = floor(P_k / M).
    - P'_k = P_k % M.
    """
    N, M = map(int, sys.stdin.readline().split())
    # Read A as a list of integers. Input A is 1-based, Python list A is 0-indexed A[0]..A[N-1]
    A = list(map(int, sys.stdin.readline().split())) 

    # P[k] stores prefix sum Sum_{i=1..k} A_i (using 1-based notation for A_i)
    # Size N+1 to store P_0 through P_N.
    P = [0] * (N + 1)       
    # P_mod[k] stores P[k] % M
    P_mod = [0] * (N + 1)   
    # q[k] stores P[k] // M (floor division)
    q = [0] * (N + 1)       
    
    # Calculate Prefix sums, P_mod (prefix sums modulo M), and q (prefix sums divided by M) values
    for i in range(N):
        # P[i+1] = P[i] + A[i] (where A[i] is A_{i+1} in 1-based problem notation)
        P[i+1] = P[i] + A[i]
        # Calculate quotient q and remainder P_mod for P[i+1] / M
        q[i+1] = P[i+1] // M
        P_mod[i+1] = P[i+1] % M

    # Calculate S_total = Sum_{1<=l<=r<=N} Sum_{i=l..r} A_i
    # This is the sum of all subarray sums. It can be calculated efficiently by summing
    # the contributions of each element A_k across all subarrays it belongs to.
    S_total = 0
    for k in range(N): # k is the 0-based index for A. A[k] corresponds to element A_{k+1}.
        # The element A_{k+1} (which is A[k]) is included in subarrays (l, r) where 1 <= l <= k+1 and k+1 <= r <= N.
        # Number of choices for start index l: k+1.
        # Number of choices for end index r: N - (k+1) + 1 = N - k.
        # Contribution of A[k] is A[k] * (number of subarrays it's in) = A[k] * (k+1) * (N-k).
        S_total += A[k] * (k + 1) * (N - k)

    # Calculate Q_sum = Sum_{r=1..N, j=0..r-1} (q_r - q_j)
    # This sum arises from the formula manipulation involving floor division.
    # It simplifies algebraically to N*q[N] + Sum_{r=1..N-1} (2*r - N) * q[r]
    Q_sum = N * q[N]
    for r in range(1, N): # The summation index r runs from 1 up to N-1
        Q_sum += (2 * r - N) * q[r]

    # Calculate Inv = Sum_{r=1..N, j=0..r-1} [P'_r < P'_j] using Fenwick Tree
    # where [condition] is 1 if the condition is true, and 0 otherwise.
    # This counts the number of 'inversions' based on the sequence of prefix sums modulo M.
    # The Fenwick Tree will store frequencies of P_mod values (which range from 0 to M-1).
    # Values 0..M-1 map to indices 1..M in the Fenwick tree. Tree needs size M.
    bit = FenwickTree(M) 

    Inv = 0
    
    # Initialize the Fenwick Tree with the first prefix sum P'_0 = P_mod[0] = 0.
    # The value 0 maps to index 1 in the 1-based Fenwick tree.
    bit.add(P_mod[0] + 1, 1) 

    for r in range(1, N + 1): # Loop through prefix sums from P_1 to P_N
        # Current prefix sum modulo M is P'_r = P_mod[r]
        
        # We need to count the number of indices j < r such that P'_j > P'_r.
        # This is equivalent to querying the sum of frequencies in the Fenwick Tree
        # for values k strictly greater than P_mod[r].
        # These values k map to indices k+1. We need indices > P_mod[r]+1.
        # The range of indices is [P_mod[r]+2, M].
        # Sum of frequencies in range [a, b] is query(b) - query(a-1).
        # Here, a = P_mod[r]+2, b = M.
        # Sum = bit.query(M) - bit.query((P_mod[r]+2) - 1) = bit.query(M) - bit.query(P_mod[r]+1).
        # bit.query(M) gives the total count of elements added so far (which is r).
        # bit.query(P_mod[r]+1) gives the count of elements with value <= P_mod[r].
        # Their difference is the count of elements with value > P_mod[r].
        
        count_greater = bit.query(M) - bit.query(P_mod[r] + 1)
        Inv += count_greater
        
        # Add the current prefix sum modulo M, P'_r = P_mod[r], to the Fenwick Tree.
        # It maps to index P_mod[r]+1. We increment its frequency count by 1.
        bit.add(P_mod[r] + 1, 1)

    # The final answer is calculated using the derived formula: T = S_total - M * (Q_sum - Inv)
    # Which is algebraically equivalent to T = S_total - M * Q_sum + M * Inv
    T = S_total - M * Q_sum + M * Inv
    print(T)

# Ensure the script runs the solve function when executed, e.g. `python script.py < input.txt`
if __name__ == '__main__':
    solve()