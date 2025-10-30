import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # pos[val] stores the 0-based index of 'val' in array A.
    # We use N+1 size for values 1 to N.
    pos = [0] * (N + 1)
    for i in range(N):
        pos[A[i]] = i

    swaps = []

    # Iterate from k=1 to N. We want to place value 'k' at index 'k-1'.
    for k in range(1, N + 1):
        target_idx_for_k = k - 1  # 0-based target index for value k
        current_idx_of_k = pos[k] # 0-based current index of value k

        # If 'k' is already in its correct position, no swap is needed.
        if current_idx_of_k == target_idx_for_k:
            continue

        # Get the value that is currently at the target position for 'k'.
        # This value needs to be moved to where 'k' currently is.
        val_at_target_idx = A[target_idx_for_k]

        # Perform the swap in the array A.
        # A[target_idx_for_k] will now hold 'k'.
        # A[current_idx_of_k] will now hold 'val_at_target_idx'.
        A[target_idx_for_k], A[current_idx_of_k] = A[current_idx_of_k], A[target_idx_for_k]

        # Update the position tracking array 'pos'.
        # 'k' is now at target_idx_for_k.
        pos[k] = target_idx_for_k
        # 'val_at_target_idx' is now at current_idx_of_k.
        pos[val_at_target_idx] = current_idx_of_k

        # Record the swap. Indices must be 1-based for output.
        # Since target_idx_for_k is always k-1 and current_idx_of_k >= k-1,
        # (target_idx_for_k + 1) will always be less than or equal to (current_idx_of_k + 1).
        # If they are equal, no swap is performed. So, i < j condition is naturally met.
        swaps.append((target_idx_for_k + 1, current_idx_of_k + 1))

    # Print the total number of swaps.
    sys.stdout.write(str(len(swaps)) + "
")
    # Print each swap operation.
    for s_i, s_j in swaps:
        sys.stdout.write(f"{s_i} {s_j}
")

# Call the solve function to execute the program.
solve()