import sys

# Fenwick Tree (Binary Indexed Tree)
class FenwickTree:
    def __init__(self, size):
        # size is the number of elements in the conceptual array (N-1 for diff)
        # ft array is 1-indexed, so size + 1
        self.size = size
        self.ft = [0] * (size + 1)

    def update(self, idx, delta):
        # idx is 1-based for the conceptual array element
        # Corresponds to conceptual array index idx - 1
        # Check bounds (optional if caller guarantees correctness)
        # if idx <= 0 or idx > self.size:
        #      return

        while idx <= self.size:
            self.ft[idx] += delta
            idx += idx & (-idx) # Move to the next node

    def query(self, idx):
        # idx is 1-based for the conceptual array, returns sum up to index idx-1
        # Corresponds to sum of conceptual array elements from index 0 to idx-1
        # Check bounds
        if idx <= 0:
             return 0 # Sum before index 1 is 0

        # if idx > self.size:
        #      idx = self.size # Sum up to the last element (or clamp)

        total = 0
        while idx > 0:
            total += self.ft[idx]
            idx -= idx & (-idx) # Move to the parent node
        return total

    def range_query(self, l, r):
        # l, r are 1-based for the conceptual array elements we want to sum
        # Corresponds to sum of conceptual array elements from index l-1 to r-1
        # Range is inclusive [l, r] in 1-based FT indices
        if l > r: # Handle empty range gracefully
             return 0

        # The caller provides l, r as 1-based indices into the conceptual array range.
        # e.g., range_query(1, N-1) sums diff[0..N-2].
        # query(r) gets sum up to conceptual index r-1.
        # query(l-1) gets sum up to conceptual index l-2.
        # Difference is sum of conceptual indices [l-1, r-1]. This is correct.

        return self.query(r) - self.query(l - 1)

# Main logic
def solve():
    # Use faster I/O
    input = sys.stdin.readline

    n, q = map(int, input().split())
    s_str = input().strip()

    # diff array: diff[i] = 1 if S[i] == S[i+1], 0 otherwise. 0-indexed, size N-1.
    # diff indices 0 to N-2.
    # This array needs to be maintained to know the current value at a diff index
    # when applying a type 1 flip.
    # If N=1, diff size is 0.
    diff = [0] * (n - 1)

    # Initialize diff based on the initial string S
    # Only meaningful for N > 1. If N=1, range(0) is empty.
    if n > 1:
        # Compare s_str[i] and s_str[i+1]
        for i in range(n - 1):
            if s_str[i] == s_str[i+1]:
                diff[i] = 1

    # Fenwick Tree for diff array. Conceptual size N-1. FT indices 1 to N-1.
    # Maps diff index i (0-based, 0..N-2) to FT index i+1 (1-based, 1..N-1).
    # If N=1, conceptual size is 0. FT size is 1. update/query should handle this.
    bit = FenwickTree(n - 1)

    # Populate Fenwick Tree based on initial diff
    # Only meaningful for N > 1. If N=1, range(0) is empty.
    if n > 1:
        for i in range(n - 1):
            if diff[i] == 1:
                bit.update(i + 1, 1) # diff[i] (0-based) corresponds to FT index i+1 (1-based)


    # Process queries
    for _ in range(q):
        query = list(map(int, input().split()))
        type = query[0]
        l, r = query[1], query[2] # l, r are 1-indexed on the original string S

        if type == 1:
            # Flip S[l-1 .. r-1] (0-indexed string indices)
            # This operation affects diff values at 0-indexed positions:
            #   - l-2 (if l-1 > 0, i.e., l > 1): compares S[l-2] and S[l-1]
            #   - r-1 (if r-1 < N-1, i.e., r < N): compares S[r-1] and S[r]

            # Check diff index l-2 (0-indexed). Relevant if l >= 2.
            # Corresponds to FT index (l-2) + 1 = l-1.
            if l >= 2:
                diff_idx = l - 2 # 0-indexed diff index
                # Flip the value at diff[diff_idx]
                # If diff[diff_idx] was 0, it becomes 1 (delta = +1)
                # If diff[diff_idx] was 1, it becomes 0 (delta = -1)
                delta = 1 - 2 * diff[diff_idx]
                diff[diff_idx] = 1 - diff[diff_idx] # Update diff array value
                bit.update(diff_idx + 1, delta) # Update FT at 1-based index (diff_idx + 1)

            # Check diff index r-1 (0-indexed). Relevant if r <= N - 1.
            # Corresponds to FT index (r-1) + 1 = r.
            if r <= n - 1:
                diff_idx = r - 1 # 0-indexed diff index
                 # Flip the value at diff[diff_idx]
                delta = 1 - 2 * diff[diff_idx]
                diff[diff_idx] = 1 - diff[diff_idx] # Update diff array value
                bit.update(diff_idx + 1, delta) # Update FT at 1-based index (diff_idx + 1)


        elif type == 2:
            # Check if substring S[l-1 .. r-1] (0-indexed string indices) is good.
            # Substring length is r - l + 1.
            # A single character string is always good.
            if r - l + 1 == 1:
                print("Yes")
            else:
                # Substring is good if S[i] != S[i+1] for all i in [l-1, r-2] (0-indexed string indices).
                # This means diff[i] must be 0 for all i in [l-1, r-2] (0-indexed diff indices).
                # Check if the sum of diff[i] in the range [l-1, r-2] is 0.

                # diff index i (0-based, 0..N-2) corresponds to FT index i+1 (1-based, 1..N-1).
                # The range of diff indices [l-1, r-2] (0-based) corresponds to
                # the range of FT indices [(l-1)+1, (r-2)+1] = [l, r-1] (1-based).

                # The FT range query needs 1-based indices [l_ft, r_ft] corresponding to
                # the conceptual array elements.
                l_ft = l
                r_ft = r - 1

                # The range [l_ft, r_ft] = [l, r-1] is valid for the FT conceptual size (N-1)
                # when N >= 2 and R-L+1 > 1, as checked before.
                # When N=1, R-L+1 is always 1, handled by the if above.
                # When N >= 2 and R-L+1 > 1, l >= 1 and r-1 >= l >= 1. r-1 <= N-1.
                # So the range [l, r-1] is contained within [1, N-1].

                total_bad_pairs = bit.range_query(l_ft, r_ft)

                if total_bad_pairs == 0:
                    print("Yes")
                else:
                    print("No")

# Execute the solver
solve()