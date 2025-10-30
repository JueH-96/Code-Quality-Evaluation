import sys

# The maximum recursion depth needed for N=12 is 12.
# Default recursion limits are usually higher (e.g., 1000), so
# increasing it might not be strictly necessary, but it's safer
# for competitive programming environments with lower default limits.
# sys.setrecursionlimit(2000) # Example: increase limit if needed

# Function to generate Restricted Growth Strings (RGS) representing partitions
# and calculate XOR sums for each partition.
def generate_rgs_and_calculate_xor(k, block_ids, max_block_id, A, possible_xor_sums):
    """
    Recursively generates RGS for elements from index k to N-1.
    When k reaches N, a complete RGS (and partition) is formed,
    and the corresponding XOR sum is calculated and added to possible_xor_sums.

    Args:
        k: The current element index (0 to N-1) to assign a block ID.
           The recursion starts with k=1, assuming element 0 is in block 0.
        block_ids: A list of size N, where block_ids[i] stores the block ID
                   assigned to element i. This list is modified during recursion.
        max_block_id: The maximum block ID used for elements 0 to k-1.
        A: The list of initial stone counts A_0, ..., A_{N-1}.
        possible_xor_sums: A set to store the calculated XOR sums.
    """
    N = len(A) # Total number of elements

    # Base case: all elements (0 to N-1) have been assigned a block ID
    if k == N:
        # Construct the partition from block_ids
        # We use a dict to group indices by their block_id
        partition_map = {}
        for i, block_id in enumerate(block_ids):
            if block_id not in partition_map:
                partition_map[block_id] = []
            partition_map[block_id].append(i)

        # Calculate the XOR sum for this partition
        current_xor_sum = 0
        # Iterate over the lists of indices for each block in the partition
        # The keys of partition_map are the block_ids, which are 0..max_block_id
        # The order of iteration over values doesn't matter for XOR
        for block_indices in partition_map.values():
             subset_sum = 0
             for index in block_indices:
                 subset_sum += A[index]
             current_xor_sum ^= subset_sum

        # Add the calculated XOR sum to the set
        possible_xor_sums.add(current_xor_sum)
        return

    # Recursive step: Assign block ID for element k

    # Option 1: Assign element k to an existing block (from 0 to max_block_id)
    for block_id in range(max_block_id + 1):
        block_ids[k] = block_id
        generate_rgs_and_calculate_xor(k + 1, block_ids, max_block_id, A, possible_xor_sums)

    # Option 2: Assign element k to a new block (max_block_id + 1)
    # A new block can be created if the new block ID (max_block_id + 1)
    # is less than the total number of allowed block IDs (which is N).
    # The maximum number of blocks in a partition of N elements is N (singletons).
    # The block IDs used are 0, 1, ..., max_block_id. The number of blocks currently is max_block_id + 1.
    # If we add a new block, its ID is max_block_id + 1, and the number of blocks
    # used will become max_block_id + 2. The maximum number of blocks in a partition is N.
    # So, a new block can be added only if max_block_id + 1 < N.
    if max_block_id + 1 < N:
         block_ids[k] = max_block_id + 1
         generate_rgs_and_calculate_xor(k + 1, block_ids, max_block_id + 1, A, possible_xor_sums)

# Main execution block
if __name__ == "__main__":
    # Read input
    # Use readline and split for potentially faster input, especially with larger inputs
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Set to store possible XOR sums
    possible_xor_sums = set()

    # Initialize block_ids list. For N >= 1, the first element (index 0) is always in block 0.
    # We start by fixing block_ids[0] = 0 and recursively assign block IDs from index 1 onwards.
    # This is the standard way to generate RGS where the first element is always in the first block.
    block_ids = [0] * N

    # The problem guarantees N >= 2, so we don't need to handle N=0 or N=1 separately.
    # Start the recursive generation from the second element (index 1).
    # The first element (index 0) is fixed in block 0, so max_block_id seen so far is 0.
    generate_rgs_and_calculate_xor(1, block_ids, 0, A, possible_xor_sums)


    # Print the number of different possible values
    print(len(possible_xor_sums))