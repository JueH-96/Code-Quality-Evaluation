import sys

def main():
    """
    This program solves the described problem by processing queries offline.
    It uses coordinate compression and a Fenwick tree (BIT) to efficiently
    calculate the length of the longest strictly increasing subsequence
    under the given constraints for each query.
    """
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    A = list(map(int, input().split()))

    queries = []
    all_values = set(A)
    for i in range(Q):
        R, X = map(int, input().split())
        queries.append((R, X, i))
        all_values.add(X)

    # Coordinate Compression
    sorted_values = sorted(list(all_values))
    val_to_comp = {val: i + 1 for i, val in enumerate(sorted_values)}
    V = len(sorted_values)

    # Group queries by R (using 0-based index R-1)
    queries_by_R = [[] for _ in range(N)]
    for R, X, q_idx in queries:
        comp_X = val_to_comp[X]
        queries_by_R[R - 1].append((comp_X, q_idx))

    # Fenwick Tree (BIT) for Range Maximum Query
    bit = [0] * (V + 1)

    def update(idx, val):
        """Updates the BIT at index idx with the maximum value."""
        while idx <= V:
            bit[idx] = max(bit[idx], val)
            idx += idx & -idx

    def query(idx):
        """Queries the maximum value in the BIT for the range [1, idx]."""
        res = 0
        while idx > 0:
            res = max(res, bit[idx])
            idx -= idx & -idx
        return res

    # Process elements of A and answer queries
    ans = [0] * Q
    for i in range(N):
        a_val = A[i]
        comp_a = val_to_comp[a_val]

        # Find max LIS length for elements strictly smaller than a_val
        lis_len_before = query(comp_a - 1)
        new_len = lis_len_before + 1

        # Update the max LIS length for elements with value a_val
        update(comp_a, new_len)

        # Answer queries for R = i + 1
        for comp_X, q_idx in queries_by_R[i]:
            ans[q_idx] = query(comp_X)

    # Print all answers
    for val in ans:
        sys.stdout.write(str(val) + '
')

if __name__ == "__main__":
    main()