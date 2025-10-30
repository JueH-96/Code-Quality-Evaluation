# YOUR CODE HERE
import sys
import bisect

# Fenwick tree (BIT) implementation
# Uses 1-based indexing for the BIT array `bit`.
# The indices correspond to the sorted unique coordinate values.
def update(bit, i, delta, M):
    """ 
    Adds delta to the element corresponding to index i in the conceptual array
    represented by the BIT. M is the total number of unique coordinates.
    The BIT array `bit` has size M+1. Valid indices are 1 to M.
    """
    # Start update from index i
    while i <= M:
        # Add delta to current BIT node
        bit[i] += delta
        # Move to the next index in the BIT that is affected by the update at index i.
        # This is done by adding the least significant bit (LSB) of i to i.
        # `i & (-i)` isolates the LSB of i.
        i += i & (-i) 

def query(bit, i):
    """ 
    Computes the prefix sum up to index i. This corresponds to the sum of counts
    for coordinates coords[0]...coords[i-1] in the original sorted unique coordinate list.
    Valid indices for query are 0 to M. query(0) returns 0.
    """
    s = 0
    # Start from index i
    while i > 0:
        # Add the value of the current BIT node to the sum
        s += bit[i]
        # Move to the 'parent' index in the BIT structure responsible for the prefix sum.
        # This is done by subtracting the LSB of i from i.
        i -= i & (-i) 
    return s

def solve():
    # Read problem parameters N (number of ants) and T (time duration)
    N, T = map(int, sys.stdin.readline().split())
    # Read the direction string S
    S = sys.stdin.readline().strip()
    # Read the initial positions X
    X = list(map(int, sys.stdin.readline().split()))

    # Store ant data. Each element contains initial position 'x' and direction 's'.
    # Direction s=0 means moving left (negative), s=1 means moving right (positive).
    ants = []
    for i in range(N):
        ants.append({'x': X[i], 's': int(S[i])})

    # Sort ants based on their initial positions X_i in ascending order.
    # This allows processing ants from left to right.
    ants.sort(key=lambda ant: ant['x'])

    # --- Coordinate Compression Step ---
    # Identify all relevant coordinate values for the problem.
    # These include the initial positions `ants[i]['x']` of all ants.
    # Also include the threshold positions `ants[i]['x'] - 2*T` which are used
    # in the crossing condition check `X_k >= X_l - 2*T`.
    coords_set = set()
    for i in range(N):
        coords_set.add(ants[i]['x'])
        coords_set.add(ants[i]['x'] - 2 * T) 

    # Create a sorted list of unique coordinate values.
    coords = sorted(list(coords_set))
    
    # Create a mapping from each unique coordinate value to its 1-based index
    # in the sorted list `coords`. This mapping is used for BIT operations.
    coord_map = {val: i + 1 for i, val in enumerate(coords)}
    
    # M stores the total number of unique coordinate values.
    # The BIT will operate on indices from 1 to M.
    M = len(coords)

    # --- Fenwick Tree (BIT) Initialization ---
    # Initialize BIT array of size M+1 with all zeros.
    bit = [0] * (M + 1)

    # --- Counting Crossings ---
    crossings = 0 # Initialize total crossing count
    total_ones_so_far = 0 # Keep track of the number of ants moving right (s=1) encountered so far

    # Iterate through the ants, already sorted by their initial position X_i.
    for i in range(N):
        x = ants[i]['x'] # Current ant's initial position
        s = ants[i]['s'] # Current ant's direction (0 or 1)
        
        if s == 0: # If the current ant moves left (s=0)
            # This ant (at sorted index i) might cross with ants k < i that move right (s_k=1).
            # A crossing occurs between ant k (position X_k, direction s_k=1) and ant i (position X_i, direction s_i=0)
            # if X_i > X_k (which is true since ants are sorted and i > k) AND they meet by time T.
            # The meeting condition is X_i - X_k <= 2*T, which implies X_k >= X_i - 2*T.
            
            # Calculate the threshold initial position for potential crossing partners (ants k < i with s_k=1).
            target_val = x - 2 * T 
            
            # Find the index `idx_in_coords` in the sorted `coords` list such that 
            # `coords[idx_in_coords]` is the first element greater than or equal to `target_val`.
            # `bisect_left` efficiently finds this index.
            idx_in_coords = bisect.bisect_left(coords, target_val)
            
            # We need to count the number of ants k < i such that s_k = 1 and X_k >= target_val.
            # The BIT stores counts of ants with s=1 at indices corresponding to their positions X_k.
            # The count needed is the sum of counts in the BIT for indices corresponding to coordinates >= target_val.
            # These coordinates are `coords[idx_in_coords]` through `coords[M-1]`.
            # Their corresponding 1-based BIT indices are `idx_in_coords + 1` through `M`.
            # The sum over this range is obtained by: (Total sum in BIT) - (Sum up to index idx_in_coords).
            # Total sum of s=1 ants seen so far is `total_ones_so_far`.
            # Sum up to index idx_in_coords (exclusive of `coords[idx_in_coords]`) is `query(bit, idx_in_coords)`.
            
            # Calculate the count of s=1 ants encountered so far whose positions X_k are strictly less than target_val.
            count_strictly_less_than_target = query(bit, idx_in_coords) 
            
            # The number of s=1 ants with positions X_k >= target_val is the total number of s=1 ants
            # minus those with positions strictly less than target_val.
            count = total_ones_so_far - count_strictly_less_than_target
            
            # Add this count to the total number of crossings found.
            crossings += count
            
        else: # s == 1, If the current ant moves right (s=1)
            # This ant could potentially cross with ants j > i that move left (s_j = 0).
            # We record this ant's position in the BIT so it can be counted by future left-moving ants.
            
            # Get the 1-based index for the BIT corresponding to this ant's initial position x.
            bit_idx = coord_map[x] 
            # Update the BIT by adding 1 at this index, signifying an ant moving right at this position.
            update(bit, bit_idx, 1, M) 
            # Increment the total count of right-moving ants encountered so far.
            total_ones_so_far += 1

    # After iterating through all ants, `crossings` holds the total count of pairs that pass each other.
    # Print the final result.
    print(crossings)

# Execute the main solver function when the script is run.
solve()