# YOUR CODE HERE
import sys

def solve():
    # Read N, X, K for the current test case
    N, X, K = map(int, sys.stdin.readline().split())

    count = 0

    # Case 1: Vertex at distance K is X itself
    # This happens only when K = 0.
    if K == 0:
        count = 1
        print(count)
        return # Problem solved for this test case

    # For K > 0, the target vertex v is different from X.

    # Calculate the depth of X relative to root 1 (at depth 0)
    # depth(i) = i.bit_length() - 1 for i >= 1
    d_X = X.bit_length() - 1

    # Case 2: Vertex is an ancestor of X at distance K
    # An ancestor k_up steps up from X is X // (2**k_up).
    # We are looking for the ancestor at k_up = K steps up.
    # This ancestor exists and is a valid node (>= 1) if K <= d_X.
    if K <= d_X:
        # The ancestor X // (2**K) exists and is >= 1 since X >= 1 and K <= d_X implies 2**K <= X
        # Note: X // (2**K) is guaranteed to be >= 1 if X >= 1 and 2**K <= X.
        # Since K <= d_X = floor(log2(X)), 2^K <= 2^floor(log2(X)) <= X.
        count += 1

    # Case 3: Vertex is a descendant of X at distance K
    # Descendants of X at distance K form a range of nodes in the full binary tree.
    # In a full tree rooted at 1, nodes at distance k down from node p are in range [p * 2^k, p * 2^k + 2^k - 1].
    # For descendants of X at distance K, the range is [X * 2^K, X * 2^K + 2^K - 1].
    # We need to count how many nodes in this range are <= N.
    # Use a threshold for the exponent K, because if 2^K exceeds N/X, the lower bound X * 2^K will exceed N.
    # Since N <= 10^18, log2(N) is approximately 60. If K >= 60, 2^K > 10^18 >= N. If X >= 1, X * 2^K > N.
    EXP_THRESHOLD = 60 # Using 60 as a safe threshold since 2^60 > 10^18

    # Check if 2^K might make the lower bound exceed N
    if K < EXP_THRESHOLD:
        # Calculate 2^K efficiently using bit shift
        pow2K = 1 << K
        lower_bound_descendants = X * pow2K

        # If the starting node of the range is already greater than N, no nodes exist in this range.
        if lower_bound_descendants <= N:
            # The range in a full tree is [lower_bound_descendants, lower_bound_descendants + pow2K - 1]
            upper_bound_full_descendants = lower_bound_descendants + pow2K - 1
            # The actual nodes are in the range [lower_bound_descendants, min(upper_bound_full_descendants, N)]
            # The number of nodes is min(upper_bound_full_descendants, N) - lower_bound_descendants + 1
            count += min(upper_bound_full_descendants, N) - lower_bound_descendants + 1
    # Else (K >= EXP_THRESHOLD): pow2K = 2^K is very large.
    # Since X >= 1 and K >= EXP_THRESHOLD (>= 60), 2^K > 10^18 >= N.
    # lower_bound_descendants = X * 2^K >= 1 * 2^K > N. Count is 0.


    # Case 4: Vertex v where LCA(X, v) = u, and u is a proper ancestor of X.
    # The path from X to v goes up from X to u, then down from u to v.
    # dist(X, u) + dist(u, v) = K.
    # Let dist(X, u) = k_up. u is the ancestor k_up steps up from X. u = X // (2**k_up).
    # k_up ranges from 1 (parent of X) up to d_X (root 1).
    # dist(u, v) = K - k_up. Let this be k_down.
    # v must be a proper descendant of u, so k_down >= 1, which means K - k_up >= 1, or k_up <= K - 1.
    # So k_up ranges from 1 up to min(d_X, K - 1).

    # Also, v is in the subtree of u, but not in the subtree of X.
    # This implies that the path from u to v must start by taking the child of u
    # which is NOT on the path from u down to X.
    # Let this child be c_other. dist(u, c_other) = 1.
    # The distance from c_other to v is dist(u, v) - dist(u, c_other) = k_down - 1 = K - k_up - 1.
    # Let k_prime = K - k_up - 1. We need k_prime >= 0, which is guaranteed by k_up <= K - 1.

    # Iterate through possible values of k_up: 1 up to d_X.
    for k_up in range(1, d_X + 1):
        # If k_up is already K or more, then k_down = K - k_up will be <= 0.
        # We need k_down >= 1 for v to be a proper descendant of u.
        if k_up >= K:
            break # Cannot go K-k_up >= 1 steps down from u

        # u is the ancestor k_up steps up from X
        u = X // (1 << k_up)

        # k_prime is the distance from c_other to v
        k_prime = K - k_up - 1 # Always >= 0 because k_up <= K - 1

        # Find the child of u that is on the path down to the original X.
        # This child is the ancestor of X that is k_up - 1 steps up from X.
        child_on_path_from_u_to_X = X // (1 << (k_up - 1))

        # Find c_other: the child of u not on the path to X.
        # u has children 2*u and 2*u + 1. One is child_on_path_from_u_to_X, the other is c_other.
        if child_on_path_from_u_to_X == 2 * u:
            c_other = 2 * u + 1
        else: # child_on_path_from_u_to_X == 2 * u + 1
            c_other = 2 * u

        # The node u = X // 2^k_up. Since X >= 1 and k_up <= d_X = floor(log2(X)),
        # 2^k_up <= 2^floor(log2(X)) <= X. So u >= X // X = 1. u is always >= 1.
        # c_other is 2*u or 2*u+1, so c_other is always >= 2.

        # If c_other itself is greater than N, its entire subtree is outside the tree [1, N].
        if c_other > N:
            continue # No nodes in this subtree exist. Move to the next k_up.

        # Count descendants of c_other at distance k_prime.
        # These nodes form range [c_other * 2^k_prime, c_other * 2^k_prime + 2^k_prime - 1].
        # We need to count how many nodes in this range are <= N.
        # Use a threshold for the exponent k_prime.
        # If k_prime >= 60, 2^k_prime > 10^18 >= N.
        # Since c_other >= 2, lower_bound = c_other * 2^k_prime >= 2 * 2^k_prime > N. Count is 0.
        if k_prime < EXP_THRESHOLD:
            pow2k_prime = 1 << k_prime
            lower_bound_other = c_other * pow2k_prime

            # If the starting node of the range is already greater than N, no nodes exist in this range.
            if lower_bound_other <= N:
                # Range in full tree is [lower_bound_other, lower_bound_other + pow2k_prime - 1]
                upper_bound_full_other = lower_bound_other + pow2k_prime - 1
                # The actual nodes are in the range [lower_bound_other, min(upper_bound_full_other, N)]
                # The number of nodes is min(upper_bound_full_other, N) - lower_bound_other + 1
                count += min(upper_bound_full_other, N) - lower_bound_other + 1
        # Else (k_prime >= EXP_THRESHOLD): lower_bound_other > N. Count is 0.


    print(count)


# Read the number of test cases
T = int(sys.stdin.readline())

# Process each test case
for _ in range(T):
    solve()