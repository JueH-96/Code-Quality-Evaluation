import sys

# Fast I/O
input = sys.stdin.readline

# Fenwick tree (Binary Indexed Tree) implementation
# BIT is 1-based indexed internally for slots 1 to N
def update(bit, idx, val, n):
    """Adds val to element at index idx (1-based)."""
    while idx <= n:
        bit[idx] += val
        idx += idx & -idx

# def query(bit, idx):
#     """Gets sum of elements from 1 to idx (1-based)."""
#     sum_val = 0
#     while idx > 0:
#         sum_val += bit[idx]
#         idx -= idx & -idx
#     return sum_val

def find_kth(bit, k, n):
    """Finds the 1-based index of the k-th available slot.
       Uses binary lifting approach for O(log N) time."""
    pos = 0
    # Determine the highest bit position relevant for N
    # We iterate through powers of 2 downwards from the largest power of 2
    # less than or equal to N. N.bit_length() gives number of bits,
    # so N.bit_length() - 1 is floor(log2(N)).
    
    log_n = n.bit_length()

    # Iterate through bit positions from most significant down to 0
    # This covers checking segments of size 2^p
    for p in range(log_n - 1, -1, -1):
        # Calculate the potential new position by adding 2^p to the current position
        new_pos = pos + (1 << p)
        
        # Check if the new_pos is within the bounds [1, N]
        # and if the count in the segment represented by bit[new_pos]
        # is less than the remaining k.
        # In the standard O(log N) find_kth, bit[new_pos] holds the sum of counts
        # for the range ending at new_pos in the implicit tree.
        # If bit[new_pos] < k, it means the k-th item is in a segment
        # further to the right. We move our current position (`pos`)
        # to `new_pos` and reduce `k` by the counts covered by the
        # segment ending at `new_pos`.
        if new_pos <= n and bit[new_pos] < k:
            k -= bit[new_pos]
            pos = new_pos
            
    # After the loop, 'pos' is the 0-based index of the element *before* the k-th item.
    # The 1-based index of the k-th item is pos + 1.
    return pos + 1

# Read input
N = int(input())
P = list(map(int, input().split()))

# Initialize result array (0-based indexing)
result = [0] * N

# Initialize Fenwick tree (BIT) for N slots (1-based indices 1 to N)
# The BIT will store counts of available slots.
bit = [0] * (N + 1)

# Initially, all N slots are available. Update the BIT to reflect this.
# We add 1 for each slot from 1 to N.
for i in range(1, N + 1):
    update(bit, i, 1, N)

# Process numbers from N down to 1
# The number i is placed according to P_i among the numbers 1..i.
# When processing in reverse (i=N down to 1), P_i gives the rank of i
# among the numbers {1..i} in the final array.
# The available slots at this step are exactly the slots in the final array
# that will be occupied by numbers {1..i}.
for i in range(N, 0, -1):
    # P_i is the desired rank (1-based) for number i among the currently
    # available slots (which correspond to numbers 1..i).
    p_i = P[i - 1] # P is 0-indexed from input

    # Find the index (1-based) of the p_i-th available slot among the total N slots
    # that are not yet filled by numbers > i.
    slot_1based = find_kth(bit, p_i, N)

    # The 0-based index in the result array is slot_1based - 1
    slot_0based = slot_1based - 1

    # Place the number i into the found slot
    result[slot_0based] = i

    # Mark this slot as unavailable in the BIT by reducing its count by 1.
    # The BIT index corresponds to the slot index (1-based).
    update(bit, slot_1based, -1, N)

# Output the final array
# Print elements separated by spaces
print(' '.join(map(str, result)))