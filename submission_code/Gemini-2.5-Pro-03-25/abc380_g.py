# YOUR CODE HERE
import sys

# Modular inverse function using Fermat's Little Theorem for prime modulus
def modinv(a, m):
    """
    Computes the modular multiplicative inverse of a modulo m.
    Assumes m is a prime number and a is not a multiple of m.
    Uses Fermat's Little Theorem: a^(m-2) === a^(-1) (mod m).
    """
    return pow(a, m - 2, m)

# Fenwick tree (BIT) implementation supporting point updates and range queries
class BIT:
    """
    Binary Indexed Tree (Fenwick Tree) implementation.
    Supports point updates (add value to index) and prefix sum queries.
    Uses 1-based indexing internally for tree structure.
    """
    def __init__(self, size):
        """
        Initializes BIT with a given size.
        The tree array has size + 1 elements to support 1-based indexing up to `size`.
        """
        self.tree = [0] * (size + 1)
        self.size = size

    def add(self, idx, val):
        """
        Adds `val` to the element at index `idx` (1-based).
        Updates all tree nodes responsible for `idx`.
        """
        # Check if idx is within valid range [1, size]
        if not (1 <= idx <= self.size):
             # If using 0-based index externally, convert it first.
             # Since values P_i are 1..N, indices used with BIT are 1..N, so idx >= 1.
             # This check is more for safety; typically indices passed are valid.
             # Let's assume idx is always valid based on problem constraints.
             pass 
        
        while idx <= self.size:
            self.tree[idx] += val
            # Move to the next index in the tree structure that needs update
            # This is done by adding the least significant bit (LSB)
            idx += idx & (-idx) 

    def query(self, idx):
        """
        Queries the prefix sum up to index `idx` (1-based).
        Returns the sum of elements from index 1 to `idx`.
        """
        # If index is out of bounds, cap it at max size or handle appropriately.
        if idx > self.size: 
            idx = self.size
        elif idx < 0: # Index cannot be negative for prefix sum
            return 0 

        s = 0
        while idx > 0:
            s += self.tree[idx]
            # Move to the 'parent' index in the tree structure
            # This is done by subtracting the least significant bit (LSB)
            idx -= idx & (-idx) 
        return s
    
    def query_range(self, L, R):
        """
        Queries the sum of elements in the range [L, R] (1-based).
        Calculates query(R) - query(L-1).
        """
        if L > R: # If range is invalid (L > R), return 0
            return 0
        # L should be at least 1 for 1-based indexing
        # if L <= 0: 
        #      L = 1 # Adjust L if necessary, but typically L >= 1 for values 1..N
        
        res_R = self.query(R)
        res_L_minus_1 = self.query(L - 1)
        # The result is the difference. No modulo needed here as BIT stores counts/raw sums.
        return res_R - res_L_minus_1

# Main function to solve the problem
def solve():
    # Read N and K from input
    N, K = map(int, sys.stdin.readline().split())
    # Read permutation P. P contains values from 1 to N. Indices are 0..N-1
    P = list(map(int, sys.stdin.readline().split()))
    # Define the modulus
    MOD = 998244353

    # --- Part 1: Calculate initial inversion count I_0 ---
    # Initialize a BIT of size N to handle values from 1 to N.
    bit_total = BIT(N)
    I_0 = 0
    # Iterate through the permutation P
    for j in range(N):
        val = P[j] # The value at index j (0-based) is P[j]
        
        # To count inversions involving P[j], we count elements P[l] seen so far (l < j)
        # such that P[l] > P[j]. This is equivalent to querying the count of values
        # in the range [val + 1, N] currently present in the BIT.
        I_0 = (I_0 + bit_total.query_range(val + 1, N)) % MOD
        
        # Add the current value P[j] to the BIT, marking it as "seen".
        bit_total.add(val, 1)

    # Handle edge case K=1: If K=1, shuffling a window of size 1 does nothing.
    # The expected inversion count is simply the initial count I_0.
    if K == 1:
         print(I_0)
         return

    # --- Part 2: Calculate sum S = sum_{i=0}^{N-K} inv_in(i) ---
    # inv_in(i) is the number of inversions within the window P[i..i+K-1]
    
    # Initialize a BIT for the sliding window calculation. Size N for values 1..N.
    bit_window = BIT(N) 
    inv_curr = 0 # Stores inv_in(i) for the window starting at index i

    # Initialize state: Calculate inv_in(0) for the first window P[0..K-1]
    for j in range(K):
        val = P[j]
        # Count inversions formed by P[j] with elements P[0..j-1] already in the window
        inv_curr = (inv_curr + bit_window.query_range(val + 1, N)) % MOD
        # Add P[j] to the window BIT
        bit_window.add(val, 1)
    
    # S accumulates the sum: inv_in(0) + inv_in(1) + ... + inv_in(N-K)
    # Initialize S with inv_in(0). Ensure it's within modulo M.
    S = inv_curr % MOD 

    # Slide the window across the permutation
    # The loop runs from i = 0 to N-K-1. In iteration i, we compute inv_in(i+1) based on inv_in(i).
    for i in range(N - K):
        # Window slides from [i, i+K-1] to [i+1, i+K]
        
        P_leave = P[i]      # Element leaving the window from the left (at index i)
        P_enter = P[i+K]    # Element entering the window from the right (at index i+K)
        
        # Calculate Term 1: Number of inversions lost due to P_leave removal.
        # These are pairs (P_leave, P_v) where i < v <= i+K-1 and P_leave > P_v.
        # This count is equivalent to the number of elements P_v in {P[i+1], ..., P[i+K-1]}
        # such that P_v < P_leave.
        # The BIT currently represents {P[i], ..., P[i+K-1]}. 
        # Querying `bit_window.query(P_leave - 1)` counts elements P_x in this window with P_x < P_leave.
        # This count correctly corresponds to sum_{v=i+1}^{i+K-1} [P_leave > P_v].
        term1 = bit_window.query(P_leave - 1)

        # Update BIT by removing P_leave's contribution (decrement count at value P_leave)
        bit_window.add(P_leave, -1)

        # Calculate Term 2: Number of inversions gained due to P_enter addition.
        # These are pairs (P_u, P_enter) where i+1 <= u <= i+K-1 and P_u > P_enter.
        # This count is equivalent to the number of elements P_u in {P[i+1], ..., P[i+K-1]}
        # such that P_u > P_enter.
        # The BIT now represents {P[i+1], ..., P[i+K-1]}.
        # Querying `bit_window.query_range(P_enter + 1, N)` counts elements P_x in this set with P_x > P_enter.
        term2 = bit_window.query_range(P_enter + 1, N)
        
        # Update the inversion count for the next window (inv_in(i+1))
        # `inv_curr` transitions from inv_in(i) to inv_in(i+1)
        # Add MOD before taking modulo to handle potential negative intermediate results
        inv_curr = (inv_curr - term1 + term2 + MOD) % MOD
        
        # Update BIT by adding P_enter's contribution (increment count at value P_enter)
        bit_window.add(P_enter, 1)
        
        # Add the inversion count of the new window (inv_in(i+1)) to the total sum S
        S = (S + inv_curr) % MOD

    # --- Part 3: Calculate the final expected value ---
    # The derived formula for expected value E is:
    # E = I_0 + K*(K-1)/4 - S / (N-K+1)  (all operations modulo M)
    
    # Calculate the term K*(K-1)/4 mod M
    K_term_numerator = (K * (K - 1)) % MOD
    # Modular inverse of 4 is needed.
    Inv4 = modinv(4, MOD)
    K_term = (K_term_numerator * Inv4) % MOD
    
    # Calculate the term S / (N-K+1) mod M
    # Need modular inverse of (N-K+1). This is always positive since N >= K >= 1.
    InvN_K_1 = modinv(N - K + 1, MOD)
    S_term = (S * InvN_K_1) % MOD

    # Combine the terms to get the final result
    # Add MOD before the final modulo operation to ensure the result is non-negative
    Result = (I_0 + K_term - S_term + MOD) % MOD
    
    # Print the final computed expected value
    print(Result)

# Execute the solver function when the script is run
solve()