import sys

def solve():
    N, T = map(int, sys.stdin.readline().split())
    S_chars = sys.stdin.readline().strip()
    X_coords = list(map(int, sys.stdin.readline().split()))

    ants_data = []
    for i in range(N):
        initial_pos = X_coords[i]
        direction_char = S_chars[i]
        
        final_pos = 0
        if direction_char == '1': # Moving positive
            final_pos = initial_pos + T
        else: # Moving negative
            final_pos = initial_pos - T
        
        ants_data.append({'initial_pos': initial_pos, 'final_pos': final_pos})

    # Sort ants by their initial positions
    ants_data.sort(key=lambda ant: ant['initial_pos'])

    # Extract the sequence of final positions. These correspond to ants
    # already sorted by their initial positions.
    final_pos_sequence = [ant['final_pos'] for ant in ants_data]

    # Coordinate compression for final_pos_sequence
    # Get unique sorted values to map them to ranks 1...M
    all_final_pos_unique_sorted = sorted(list(set(final_pos_sequence)))
    
    rank_map = {val: i + 1 for i, val in enumerate(all_final_pos_unique_sorted)}
    
    M = len(all_final_pos_unique_sorted) # Max rank, also size of BIT array needed
    
    # Fenwick tree (BIT)
    # Array is 1-indexed for easier parent/child calculations, so size M+1
    bit_array = [0] * (M + 1)

    def update_bit(idx, delta):
        # idx is 1-based rank
        while idx <= M: 
            bit_array[idx] += delta
            idx += idx & (-idx) # Move to the next index that covers this one

    def query_bit(idx):
        # idx is 1-based rank
        s = 0
        while idx > 0:
            s += bit_array[idx]
            idx -= idx & (-idx) # Move to the parent index in BIT structure
        return s

    total_crossings = 0
    
    # Iterate through ants, which are already sorted by initial_pos.
    # Their final_pos values are in final_pos_sequence.
    # For each ant_k (represented by final_pos_sequence[k]),
    # count how many ants ant_j (with j < k, meaning ant_j is to the left of ant_k initially)
    # have ant_j.final_pos >= ant_k.final_pos.
    # This is equivalent to:
    # (number of ants j < k) - (number of ants j < k with ant_j.final_pos < ant_k.final_pos)
    
    for k in range(N): # k from 0 to N-1, representing index in sorted list
        current_ant_final_pos = final_pos_sequence[k]
        rank_of_current_final_pos = rank_map[current_ant_final_pos]
        
        # Number of ants processed before this one (ant_0, ..., ant_{k-1}) is k.
        
        # query_bit(rank_of_current_final_pos - 1) counts ants (among first k processed)
        # whose final_pos is strictly less than current_ant_final_pos.
        # (ranks are 1 to M, so rank_of_current_final_pos-1 means query up to the rank just below it)
        count_strictly_less = query_bit(rank_of_current_final_pos - 1)
        
        # So, k - count_strictly_less is the number of ants (among first k processed)
        # whose final_pos is >= current_ant_final_pos.
        # These are the crossings where the current ant is the right one of the pair.
        total_crossings += k - count_strictly_less
        
        # Add current ant's final_pos (via its rank) to the BIT structure
        update_bit(rank_of_current_final_pos, 1)
        
    sys.stdout.write(str(total_crossings) + "
")

solve()