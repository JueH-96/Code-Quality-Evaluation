import sys

# Function to calculate initial inversion count using Fenwick Tree (BIT)
def calculate_inversions(arr, n):
    """
    Calculates the total number of inversions in a given array using a Fenwick Tree (BIT).
    An inversion is a pair (i, j) such that i < j and arr[i] > arr[j].
    Time complexity: O(N log N)
    """
    bit = [0] * (n + 1)
    inv_count = 0

    # Helper function to update BIT: increments value at idx by 1
    def update(idx, val):
        while idx <= n:
            bit[idx] += val
            idx += idx & (-idx)

    # Helper function to query BIT: returns sum of values from 1 to idx
    def query(idx):
        s = 0
        while idx > 0:
            s += bit[idx]
            idx -= idx & (-idx)
        return s

    # Iterate from right to left (N-1 down to 0)
    # For each element arr[i], query the BIT for numbers smaller than arr[i]
    # that have already been added (i.e., are to its right).
    # These are the elements that form an inversion with arr[i].
    for i in range(n - 1, -1, -1):
        # Count elements already processed (to the right of arr[i]) that are smaller than arr[i]
        inv_count += query(arr[i] - 1)
        # Add arr[i] to the BIT
        update(arr[i], 1)
    return inv_count

def solve():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    M = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate initial inversion count
    current_inversions = calculate_inversions(P, N)
    
    # P needs to be mutable for swaps
    P_list = list(P)
    
    # Stores the maximum 'k' value from A that has been fully processed by a bubble sort pass.
    # Initialized to 1, because operation k=2 processes indices P[0] and P[1].
    # So P[0] is the maximum index that could have been influenced by a conceptual k=1.
    last_k_processed = 1 
    
    results = []

    # Iterate through each operation specified in sequence A
    for k_val in A:
        # Optimization: Only perform a bubble sort pass if the current k_val
        # is strictly greater than the largest k_val processed so far.
        # If k_val <= last_k_processed, the array P_list is already in the state
        # it would be after an operation k_val.
        if k_val > last_k_processed:
            # Perform one pass of bubble sort on the prefix P_list[0 ... k_val-1]
            # (which corresponds to 0-indexed elements P_list[j] for j from 0 to k_val-2).
            # This loop must start from 0, not last_k_processed - 1, because elements
            # in the already processed prefix (0 to last_k_processed-2) can be
            # re-ordered by elements from the newly added part (last_k_processed-1 to k_val-2).
            for j in range(k_val - 1): 
                if P_list[j] > P_list[j+1]:
                    P_list[j], P_list[j+1] = P_list[j+1], P_list[j]
                    # Each successful swap of an adjacent inversion reduces the total inversion count by 1.
                    current_inversions -= 1
            
            # Update the highest 'k' value that has been fully processed
            last_k_processed = k_val
        
        # After applying the necessary operations (or skipping if already processed),
        # append the current inversion count to the results.
        results.append(current_inversions)

    # Print all accumulated results
    for res in results:
        sys.stdout.write(str(res) + "
")

# Call the solve function to run the program
solve()