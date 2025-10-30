import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Stores the distinct XOR sums found. A set is used to handle uniqueness automatically.
    result_sums = set()

    # sums_of_bins stores the current sum of stones for each 'bin' (part of the partition).
    # We can have at most N bins, so initialize with N zeros.
    sums_of_bins = [0] * N

    def find_partitions(idx, num_active_bins):
        """
        Recursively finds all partitions of the input array A and calculates their XOR sums.

        Args:
            idx (int): The current index of the element A[idx] being considered.
            num_active_bins (int): The number of non-empty bins (parts) currently in the partition.
                                   These bins are represented by sums_of_bins[0] through sums_of_bins[num_active_bins-1].
        """
        # Base case: All A_i elements have been placed into bins
        if idx == N:
            current_xor_sum = 0
            # Calculate the XOR sum of all active bins
            for i in range(num_active_bins):
                current_xor_sum ^= sums_of_bins[i]
            result_sums.add(current_xor_sum)
            return

        # Recursive step for A[idx]:

        # Case 1: Place A[idx] into an existing bin
        # Iterate through all currently active bins (from 0 to num_active_bins - 1)
        for j in range(num_active_bins):
            sums_of_bins[j] += A[idx]  # Add A[idx] to bin j
            find_partitions(idx + 1, num_active_bins) # Recurse for the next element
            sums_of_bins[j] -= A[idx]  # Backtrack: remove A[idx] from bin j to explore other possibilities

        # Case 2: Place A[idx] into a new bin
        # A new bin is created at the index `num_active_bins`.
        # This approach ensures each partition is generated exactly once in a canonical way.
        sums_of_bins[num_active_bins] = A[idx] # Place A[idx] into the new bin
        find_partitions(idx + 1, num_active_bins + 1) # Recurse, increasing the count of active bins
        # No explicit backtracking for sums_of_bins[num_active_bins] is needed here.
        # Its value will either be overwritten in a subsequent call (if this bin index is reused),
        # or it will be out of scope/ignored if num_active_bins reduces in parent calls.

    # Start the recursion: begin with the first element (index 0) and no active bins (0).
    find_partitions(0, 0)

    # The final answer is the total count of distinct XOR sums found.
    print(len(result_sums))

# Call the solve function to run the program
solve()