import sys
import bisect
import math

# Sparse Table implementation for Range Maximum Query (RMQ)
class SparseTable:
    def __init__(self, arr):
        self.n = len(arr)
        # Precompute log_2 values for quick lookup of block size in query
        # log_table[i] stores floor(log2(i))
        self.log_table = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.log_table[i] = self.log_table[i // 2] + 1
        
        # Determine the maximum k such that 2^k <= n
        # This determines the number of 'rows' in our sparse table.
        # For N=2e5, log2(N) is approx 17.6, so max_log_k will be 17.
        max_log_k = self.log_table[self.n]
        
        # st[k][i] stores the maximum in the range [i, i + 2^k - 1]
        # k is the power of 2, i is the starting index.
        self.st = [[0] * self.n for _ in range(max_log_k + 1)]
        
        # Base case: k=0 (interval length 2^0 = 1)
        # Each element is the maximum of itself.
        for i in range(self.n):
            self.st[0][i] = arr[i]
            
        # Fill the table for k from 1 up to max_log_k
        # Each cell st[k][i] is computed from two cells in the previous row (k-1)
        for k in range(1, max_log_k + 1):
            # Iterate through all possible starting positions i for an interval of size 2^k
            # The interval [i, i + 2^k - 1] must fit within the array [0, n-1].
            # So, i + 2^k - 1 < n  =>  i < n - 2^k + 1
            for i in range(self.n - (1 << k) + 1):
                # The maximum of [i, i + 2^k - 1] is the max of:
                # 1. The first half: [i, i + 2^(k-1) - 1] (covered by st[k-1][i])
                # 2. The second half: [i + 2^(k-1), i + 2^k - 1] (covered by st[k-1][i + 2^(k-1)])
                self.st[k][i] = max(self.st[k-1][i], self.st[k-1][i + (1 << (k-1))])
                
    def query(self, L, R): # query max in range [L, R] inclusive (0-indexed)
        if L > R: 
            # In this problem, L <= R will always hold for valid query ranges.
            # Returning -sys.maxsize is a general way to indicate an empty or invalid range.
            return -sys.maxsize 
        
        # k is the largest power of 2 such that 2^k <= (R - L + 1), the length of the query range.
        k = self.log_table[R - L + 1]
        
        # The range [L, R] is covered by two potentially overlapping blocks of size 2^k:
        # 1. Starting at L: [L, L + 2^k - 1]
        # 2. Ending at R: [R - 2^k + 1, R]
        # Taking the maximum of these two blocks covers the entire query range [L, R].
        return max(self.st[k][L], self.st[k][R - (1 << k) + 1])

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # 1. Precompute next_pair_idx[i] for i=0, ..., N-1
    # next_pair_idx[i] will store the smallest index j > i such that A[j] >= 2 * A[i].
    # If no such j exists (i.e., no mochi after A[i] is large enough), next_pair_idx[i] is N.
    next_pair_idx = [0] * N
    for i in range(N):
        # bisect_left finds the insertion point for 2 * A[i] in the sorted array A.
        # The search starts from index i+1 to ensure that the bottom mochi's index (j) is greater than
        # the top mochi's index (i), as they must be distinct mochi.
        j = bisect.bisect_left(A, 2 * A[i], lo=i + 1)
        
        # Check if a valid index `j` was found within the array bounds.
        # If `j` is less than `N`, it means `A[j]` exists.
        # The `bisect_left` guarantees `A[j] >= 2 * A[i]` (if `j < N`).
        if j < N:
            next_pair_idx[i] = j
        else:
            # If `j` is `N` (or greater), it means no element `>= 2 * A[i]` exists after index `i`.
            # In this case, A[i] cannot be a top mochi for any valid pair, so we use N as a sentinel.
            next_pair_idx[i] = N

    # 2. Create array B[i] = next_pair_idx[i] - i
    # This transformation is crucial for the RMQ step.
    # The condition `R_idx - K + 1 >= next_pair_idx[L_idx+i] - i` for all relevant `i`
    # becomes `R_idx - K + 1 >= max_{relevant_i} B[i]`.
    B = [0] * N
    for i in range(N):
        B[i] = next_pair_idx[i] - i

    # 3. Build a Sparse Table on array B for Range Maximum Query (RMQ)
    st_b = SparseTable(B)

    Q = int(sys.stdin.readline())
    results = []

    for _ in range(Q):
        L, R = map(int, sys.stdin.readline().split())
        L_idx = L - 1 # Convert 1-indexed L to 0-indexed L_idx
        R_idx = R - 1 # Convert 1-indexed R to 0-indexed R_idx

        # Binary search for K (the maximum number of kagamimochi)
        # K can range from 0 to floor((number of mochi in range) / 2)
        low = 0
        high = (R_idx - L_idx + 1) // 2
        ans = 0 # This will store the maximum K found to be possible

        while low <= high:
            mid_k = (low + high) // 2

            # If mid_k is 0, it means 0 pairs. This is always possible.
            if mid_k == 0:
                ans = max(ans, mid_k) # Update ans if 0 is the current best (e.g., first iteration)
                low = mid_k + 1       # Try to find if more pairs are possible
                continue
            
            # For mid_k pairs, the 'top' mochi candidates are A[L_idx] through A[L_idx + mid_k - 1].
            # We need to find the maximum value of B[i] = (next_pair_idx[i] - i)
            # within this range of 'top' mochi indices.
            query_start_idx = L_idx
            query_end_idx = L_idx + mid_k - 1
            
            # Perform the Range Maximum Query using the Sparse Table
            max_B_val = st_b.query(query_start_idx, query_end_idx)
            
            # The condition for 'mid_k' pairs to be possible is:
            # (Starting index of the chosen 'bottom' mochi group) >= (maximum of B[i] for 'top' mochi)
            # The starting index for the chosen 'bottom' mochi group is R_idx - mid_k + 1.
            if R_idx - mid_k + 1 >= max_B_val:
                # If possible with `mid_k` pairs, it means we can achieve at least `mid_k`.
                # We store this as a potential answer and try to find if even more pairs are possible.
                ans = mid_k
                low = mid_k + 1
            else:
                # If not possible with `mid_k` pairs, then we need to try with fewer pairs.
                high = mid_k - 1
        
        results.append(ans)
    
    # Print all the answers, each on a new line.
    sys.stdout.write("
".join(map(str, results)) + "
")

# Call the solve function to run the program
solve()