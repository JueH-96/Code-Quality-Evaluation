# YOUR CODE HERE
import sys

# Increase recursion depth if necessary, although the main logic is iterative.
# sys.setrecursionlimit(2000) 

def solve():
    # Read input N and M
    N, M = map(int, sys.stdin.readline().split())
    
    # Read N sequences of length M
    A = []
    for i in range(N):
        # Store sequences as tuples for hashability
        A.append(tuple(map(int, sys.stdin.readline().split())))

    MOD = 998244353

    # Memoization cache for the transform function
    memo_transform = {}
    
    # Function to apply the transformation T to a sequence
    def transform(seq):
        # Check cache first to avoid recomputation
        if seq in memo_transform:
            return memo_transform[seq]
        
        # Compute the transformed sequence
        new_seq = [0] * M
        current_sum = 0
        for k in range(M):
            current_sum = (current_sum + seq[k]) % 2
            new_seq[k] = current_sum
        
        res = tuple(new_seq)
        # Store the result in cache before returning
        memo_transform[seq] = res
        return res

    # Dictionary to map sequences to their orbit information: (orbit_id, position_in_orbit)
    orbit_map = {} 
    # List to store detailed information for each orbit found
    # Each element is a dictionary: {'period': p, 'members': list[(original_index_i, position_k)]}
    orbit_data = [] 
    
    orbit_count = 0 # Counter for assigning unique orbit IDs

    # Group original indices by their initial sequence value
    # This helps process identical initial sequences together
    initial_indices = {} 
    for i in range(N):
        seq = A[i]
        if seq not in initial_indices:
            initial_indices[seq] = []
        initial_indices[seq].append(i)

    # Get a list of unique initial sequences to process
    unique_initial_seqs = list(initial_indices.keys())

    # Iterate through unique initial sequences to find orbits
    for seq in unique_initial_seqs:
        
        # If this sequence has already been assigned to an orbit, we can skip it
        if seq in orbit_map: 
            continue

        # Start discovering a new orbit from this sequence
        orbit_id = orbit_count
        current_orbit_sequences = [] # Stores sequences in the order they appear in the orbit path
        current_seq = seq
        
        # Temporarily track sequences visited during this specific orbit discovery path
        # This helps detect when a cycle is closed
        temp_visited_this_orbit = {} 
        
        pos = 0 # Position counter along the path
        
        # Follow the transformation path until a sequence is encountered that is already known
        # (either from earlier in this path or from a previously found orbit)
        while current_seq not in orbit_map and current_seq not in temp_visited_this_orbit:
             temp_visited_this_orbit[current_seq] = pos
             current_orbit_sequences.append(current_seq)
             current_seq = transform(current_seq)
             pos += 1

        # Check if the loop closed by returning to a sequence within the current path
        if current_seq in temp_visited_this_orbit: 
             start_cycle_pos = temp_visited_this_orbit[current_seq]
             
             # The transformation T corresponds to multiplication by an invertible matrix M over GF(2).
             # Such transformations act as permutations on the state space.
             # Therefore, all sequences must belong to pure cycles, meaning start_cycle_pos must be 0.
             # If start_cycle_pos != 0, it would imply a transient path leading into a cycle,
             # which contradicts the properties of permutations.
             assert start_cycle_pos == 0, "Detected non-pure cycle, unexpected for a permutation."

             period = pos # The cycle length (period) is the number of steps taken (pos)
             
             current_members = [] # List to store members of this orbit: (original_index, position_k)

             # Assign orbit information (orbit_id, position_k) to all sequences in the found cycle
             for k in range(period):
                 cycle_seq = current_orbit_sequences[k]
                 orbit_map[cycle_seq] = (orbit_id, k) # Position 'k' is relative to the start sequence 'seq'
                 
                 # Check if this sequence appeared in the initial input A.
                 # If yes, record all original indices associated with it.
                 if cycle_seq in initial_indices:
                     for original_idx in initial_indices[cycle_seq]:
                         current_members.append((original_idx, k))
             
             # Store the gathered information for this orbit
             orbit_data.append({'period': period, 'members': current_members})
             orbit_count += 1 # Increment orbit counter for the next new orbit
        
        else: 
             # This case implies current_seq is already in orbit_map, meaning the path merged
             # into a previously discovered orbit. This should not happen for a permutation T.
             assert False, "Merging into existing orbit detected, unexpected for permutation T."


    total_sum = 0 # Initialize the total sum accumulator

    # Fenwick tree (Binary Indexed Tree) implementation for inversion counting
    # Note: All operations are modulo MOD to prevent overflow and keep results within field.
    # 'm' is the size of the BIT array (typically max_value + 1)
    def update(bit, i, v, m):
        i += 1 # Convert to 1-based index for BIT
        while i < m: 
            bit[i] = (bit[i] + v) % MOD
            i += i & (-i) # Move to the next index to update

    def query(bit, i):
        res = 0
        i += 1 # Convert to 1-based index for BIT
        while i > 0:
            res = (res + bit[i]) % MOD
            i -= i & (-i) # Move to the parent index
        return res

    # Process each found orbit to calculate its contribution to the total sum
    for orbit_id in range(orbit_count):
        orbit_info = orbit_data[orbit_id]
        p = orbit_info['period'] # Period of the current orbit
        members = orbit_info['members'] # List of (original_index, position_k)
        
        # If this orbit contains no members from the initial input sequences, skip it
        if not members: continue 

        # Sort members based on their original index 'i' to apply the summation formula correctly
        members.sort(key=lambda x: x[0])
        
        m = len(members) # Number of members from input A in this orbit
        # Extract the list of positions 'k' corresponding to the sorted original indices
        K = [member[1] for member in members] 

        # Calculate S_raw = sum_{s=1..m} (2s - m - 1) * K_s (mod MOD)
        # This is the sum part without considering the modulo 'p' operation inside f(i,j)
        current_orbit_sum_raw = 0
        for s in range(m):
            # Use (s+1) for the 1-based index 's' in the formula
            coeff = (2 * (s + 1) - m - 1)
            term = coeff * K[s]
            current_orbit_sum_raw = (current_orbit_sum_raw + term) % MOD
        
        # Calculate N_inv = number of inversions in the sequence K (mod MOD)
        # An inversion is a pair (s, t) such that s < t and K[s] > K[t].
        # We use a Fenwick tree for efficient calculation in O(m log p) time.
        
        # BIT size needs to accommodate indices from 0 to p-1. So size p+1.
        bit_size = p + 1 
        bit = [0] * bit_size # Initialize BIT with zeros
        N_inv = 0 # Inversion count accumulator
        count_so_far = 0 # Tracks number of elements processed so far
        
        # Iterate through the positions K = [k_{i_1}, ..., k_{i_m}]
        for k_val in K: 
             # Query BIT for count of elements seen so far with position value <= k_val
             le_count = query(bit, k_val) 
             
             # Number of elements seen so far with position value > k_val
             # is (total elements seen so far) - (count <= k_val)
             # Ensure result is non-negative modulo MOD
             greater_count = (count_so_far - le_count + MOD) % MOD 
             
             # Each such element forms an inversion with the current k_val
             N_inv = (N_inv + greater_count) % MOD 
             
             # Update BIT to include the current element k_val
             update(bit, k_val, 1, bit_size)
             count_so_far += 1

        # Calculate the total sum contributed by this orbit using the formula:
        # S_O = S_raw + p * N_inv (mod MOD)
        term_p_N_inv = ((p % MOD) * N_inv) % MOD # Calculate (p * N_inv) mod MOD first
        current_orbit_total_sum = (current_orbit_sum_raw + term_p_N_inv) % MOD
        
        # Add this orbit's contribution to the overall total sum
        total_sum = (total_sum + current_orbit_total_sum) % MOD

    # Ensure the final result is non-negative before printing
    print((total_sum + MOD) % MOD)

# Execute the main function
solve()