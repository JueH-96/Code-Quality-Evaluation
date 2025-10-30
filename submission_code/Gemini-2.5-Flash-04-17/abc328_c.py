import sys

# Read N and Q
line1 = sys.stdin.readline().split()
N = int(line1[0])
Q = int(line1[1])

# Read string S
S = sys.stdin.readline().strip()

# Create an array to mark where double characters occur
# `is_double_start_indices[i] = 1` if S[i] == S[i+1] for i = 0, ..., N-2.
# This array has size N-1.
# For the query (l_i, r_i), we are interested in indices p (1-based) from l_i to r_i - 1
# such that S_p = S_{p+1}. This corresponds to 0-based indices j = p-1 such that
# l_i - 1 <= j <= r_i - 2 and S[j] == S[j+1].
# The indices j range from 0 to N-2 where S[j] == S[j+1] can occur.
# So, `is_double_start_indices[j]` being 1 means S[j] == S[j+1].
is_double_start_indices = [0] * (N - 1)
for i in range(N - 1):
    if S[i] == S[i+1]:
        is_double_start_indices[i] = 1

# Build the prefix sum array.
# `prefix_sum[k]` stores the sum of `is_double_start_indices[0]` through `is_double_start_indices[k-1]`.
# This array will have size N.
# `prefix_sum[0] = 0` (sum of first 0 elements)
# `prefix_sum[k] = sum(is_double_start_indices[0]...is_double_start_indices[k-1])` for k = 1, ..., N-1.
prefix_sum = [0] * N
for k in range(1, N):
    # prefix_sum[k] is sum up to index k-1 in is_double_start_indices
    prefix_sum[k] = prefix_sum[k-1] + is_double_start_indices[k-1]

# For a query (l_i, r_i), we need to count p such that l_i <= p <= r_i - 1 and S_p = S_{p+1}.
# This is the sum of (1 if S_p == S_{p+1}) for p from l_i to r_i - 1 (1-based).
# This corresponds to summing `is_double_start_indices[j]` for j from l_i - 1 to r_i - 2 (0-based).
# The sum of elements in a range [a, b] in an array using prefix sums P (where P[k] = sum(arr[0]...arr[k-1]))
# is given by P[b+1] - P[a].
# Here, the array is `is_double_start_indices`, the start index a = l_i - 1, and the end index b = r_i - 2.
# The sum is prefix_sum[(r_i - 2) + 1] - prefix_sum[l_i - 1]
# which simplifies to prefix_sum[r_i - 1] - prefix_sum[l_i - 1].

# Process queries
for _ in range(Q):
    line = sys.stdin.readline().split()
    l = int(line[0])
    r = int(line[1])

    # The answer for the query (l, r) is the sum of `is_double_start_indices`
    # from index l-1 up to r-2 (inclusive).
    # Using the prefix sum: sum from index (l-1) to (r-2) is prefix_sum[(r-2)+1] - prefix_sum[l-1].
    # Note that the prefix sum indices are 0-based counts.
    # prefix_sum[k] = count of doubles S[j]==S[j+1] for j in [0, k-1].
    # Count of doubles for p in [1, k] (S_p=S_{p+1}) is also prefix_sum[k].
    # We need count for p in [l, r-1].
    # (Count for p in [1, r-1]) - (Count for p in [1, l-1])
    # = prefix_sum[r-1] - prefix_sum[l-1]

    ans = prefix_sum[r - 1] - prefix_sum[l - 1]
    print(ans)