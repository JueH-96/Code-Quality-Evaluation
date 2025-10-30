# YOUR CODE HERE
import sys

# Fenwick Tree implementation (1-based indexing)
class FenwickTree:
    """
    A Fenwick Tree (Binary Indexed Tree) supports point updates and prefix sum queries
    in O(log N) time complexity. It uses 1-based indexing internally for values/indices stored.
    """
    def __init__(self, size):
        """
        Initializes the Fenwick Tree.
        :param size: The maximum index/value that can be stored (determines the size of the internal array).
                     The internal array size will be size + 1 to accommodate 1-based indexing from 1 to size.
        """
        self.tree = [0] * (size + 1)
        self.size = size # Store the maximum index capacity

    def add(self, i, delta):
        """
        Adds 'delta' to the element count at index 'i'. This is effectively incrementing the count
        for value 'i'.
        :param i: The 1-based index (representing a value V_k) where the count should be increased.
        :param delta: The value to add (typically 1 in this problem, representing one alien having this V_k value).
        """
        # Ensure index 'i' is within the valid range [1, self.size]
        # If i exceeds capacity, cap it at the maximum index allowed. This handles large V_k values.
        if i > self.size:
            i = self.size 
        # The problem logic guarantees i >= 1, as V_k = A_k + R_k + k >= 1. Check i > 0 just in case.
        if i <= 0: 
            # This case should theoretically not happen based on problem constraints and derived logic.
            # If it did, it might indicate an issue elsewhere. We return to prevent errors.
            return

        # Update the tree by adding delta to index i and all its ancestors in the BIT structure.
        while i <= self.size:
            self.tree[i] += delta
            # Move to the next index that needs update: i + (i & -i) finds the next node 
            # in the update path (effectively goes up towards the root).
            i += i & (-i) 

    def query(self, i):
        """
        Calculates the prefix sum up to index 'i'. In this context, it gives the total count 
        of elements whose values V_k are less than or equal to 'i'.
        :param i: The 1-based index up to which the sum (count) is required.
        :return: The prefix sum Sum(counts[1...i]).
        """
        # Handle boundary cases for index 'i'
        if i <= 0:
            return 0 # Query up to index 0 or negative yields a count of 0.
        # If index exceeds capacity, query up to the maximum possible index, self.size.
        if i > self.size:
            i = self.size
            
        s = 0 # Initialize sum (count)
        # Traverse down the tree from index i to calculate the prefix sum.
        while i > 0:
            s += self.tree[i]
            # Move to the parent index covering the next lower range: i - (i & -i) finds the node 
            # responsible for the range ending just before the current node's range start.
            i -= i & (-i)
        return s

def solve():
    """
    Main function to solve the problem. Reads input, computes results using the derived logic
    and Fenwick Tree, and prints the final stone counts for each alien.
    """
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Determine the maximum possible value for V_k = A_k + R_k + k
    # Max A_k is 5e5. Max N is 5e5. Max R_k is N-1, approx 5e5.
    # Max V_k approx 5e5 + 5e5 + 5e5 = 1.5e6.
    # Set MAX_VAL slightly larger than the maximum possible V_k for safety. 2,000,001 is used.
    MAX_VAL = 2000001 
    bit = FenwickTree(MAX_VAL) # Initialize Fenwick Tree with capacity MAX_VAL

    R = [0] * N # Array to store R_k values (R[k-1] stores R_k for year k)

    # Iterate through each year k from 1 to N. Alien k becomes an adult in year k.
    for k in range(1, N + 1): 
        # Calculate R_k: the number of stones alien k receives upon becoming adult.
        # R_k equals the number of adults i < k who have at least one stone at the start of year k.
        # We derived that this is equivalent to: R_k = |{ i \in {1, ..., k-1} | V_i >= k }|
        # where V_i = A_i + R_i + i.
        # This count can be computed using the Fenwick Tree:
        # R_k = (Total count of i < k) - |{ i \in {1, ..., k-1} | V_i < k }|
        # R_k = (k-1) - |{ i \in {1, ..., k-1} | V_i <= k-1 }|
        
        current_Rk = 0 # Base case: For k=1, R_1 = 0 (no prior adults).
        if k > 1:
            # Total number of elements (aliens 1 to k-1) processed so far is k-1.
            total_elements = k - 1
            
            # Query the BIT for the count of V_i values such that V_i <= k-1.
            # The index for the query is k-1.
            query_idx = k - 1
            count_less_equal_k_minus_1 = bit.query(query_idx)
            
            # R_k is the count of elements with V_i >= k.
            current_Rk = total_elements - count_less_equal_k_minus_1
            
        R[k-1] = current_Rk # Store the calculated R_k value for alien k

        # Calculate V_k = A_k + R_k + k. This value determines when alien k stops giving stones.
        # Use 0-based index k-1 for accessing A and R arrays.
        current_Vk = A[k-1] + current_Rk + k
        
        # Ensure V_k is within the bounds representable by the Fenwick Tree [1, MAX_VAL].
        # Cap V_k if it exceeds MAX_VAL. This preserves the property that it's a large value.
        if current_Vk > MAX_VAL:
           current_Vk = MAX_VAL
        # V_k >= 1 is guaranteed since A_k>=0, R_k>=0, k>=1.
        
        # Add the count for this V_k value to the Fenwick tree.
        # This effectively registers alien k with its characteristic value V_k.
        bit.add(current_Vk, 1)

    # After computing all R_k values (stones received), calculate the final stone counts B_k for each alien k.
    B = [0] * N # Array to store the final stone counts B_k
    for k in range(1, N + 1):
        # The final stone count B_k is derived from the state after N years.
        # We found the formula: B_k = max(0, A_k + R_k - (N-k))
        # This formula accounts for initial stones A_k, received stones R_k, and the maximum 
        # number of stones alien k could potentially give away from year k+1 to N, which is N-k.
        # The max(0, ...) ensures the count doesn't become negative.
        
        # Use 0-based index k-1 for accessing A and R arrays.
        val = A[k-1] + R[k-1] - (N - k)
        B[k-1] = max(0, val) # Final stone count cannot be negative.

    # Print the resulting list B containing the final stone counts B_1, B_2, ..., B_N, separated by spaces.
    print(*(B))

# Execute the solve function to run the program
solve()