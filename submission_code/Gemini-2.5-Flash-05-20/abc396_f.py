import sys

# Fenwick Tree (BIT) implementation for 0-indexed values
class FenwickTree:
    def __init__(self, size):
        # `size` is the maximum possible value + 1 (i.e., M for values 0 to M-1).
        # BIT array is 1-indexed internally.
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, idx, delta):
        # `idx` is the 0-indexed value.
        # Convert to 1-indexed for BIT operations.
        idx += 1
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & (-idx)

    def query(self, idx):
        # Query sum from 0 to `idx` (inclusive). `idx` is 0-indexed value.
        # Convert to 1-indexed for BIT operations.
        idx += 1
        s = 0
        while idx > 0:
            s += self.tree[idx]
            idx -= idx & (-idx)
        return s

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # --- Step 1: Calculate initial inversion number for k=0 (sequence A itself) ---
    # B(0) = A
    ft = FenwickTree(M)
    current_inversions = 0
    for x_val in A:
        # To find inversions: count elements already seen that are greater than x_val.
        # These elements appear before x_val in the sequence and are larger.
        # ft.query(M-1) gives the total count of all elements seen so far (sum from 0 to M-1).
        # ft.query(x_val) gives the count of elements seen so far that are <= x_val.
        # Their difference gives the count of elements seen so far that are > x_val.
        current_inversions += ft.query(M - 1) - ft.query(x_val)
        # Add x_val to the Fenwick Tree.
        ft.update(x_val, 1)

    # --- Step 2: Precompute indices_by_value ---
    # `indices_by_value[val]` will be a list of 0-indexed positions `p` where `A[p] = val`.
    indices_by_value = [[] for _ in range(M)]
    for i in range(N):
        indices_by_value[A[i]].append(i)
    
    # --- Step 3: Iterate k from 0 to M-1 and update inversion count ---
    # Store results for each k.
    results = []

    for k in range(M):
        # Add the current inversion count (for the current k) to results.
        results.append(current_inversions)

        # Determine `value_to_wrap`: the value `X` such that `(X + k) % M = M-1`.
        # This means `B_i(k) = M-1` for `A_i = X`.
        # For the next step `k+1`, these `B_i(k+1)` values will become `0`.
        # `X = (M - 1 - k) % M`. The `+ M` is added to ensure positive result if `M-1-k` is negative,
        # though for `k` from `0` to `M-1`, `M-1-k` is always `0` or positive.
        value_to_wrap = (M - 1 - k + M) % M
        
        # Get all indices `p` where `A[p]` has this `value_to_wrap`.
        # These are the elements that transition from `M-1` to `0`.
        indices_S_k = indices_by_value[value_to_wrap]
        
        # If there are any elements that wrap around for this k:
        if indices_S_k:
            # `sum_of_p_in_Sk`: sum of 0-indexed positions of elements that wrap.
            sum_of_p_in_Sk = sum(indices_S_k)
            # `num_in_Sk`: count of elements that wrap.
            num_in_Sk = len(indices_S_k)
            
            # Calculate the change in inversion count (delta_inv) using the derived formula.
            # delta_inv = 2 * sum(p for p in S_k) - (N - 1) * len(S_k)
            delta_inv = 2 * sum_of_p_in_Sk - (N - 1) * num_in_Sk
            
            # Update the current inversion count for the next iteration (k+1).
            current_inversions += delta_inv

    # Print all calculated inversion numbers.
    for ans in results:
        sys.stdout.write(str(ans) + '
')

solve()