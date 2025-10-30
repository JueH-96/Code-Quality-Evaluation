# YOUR CODE HERE
import sys

def check(H, N, X, U, D):
    """
    Checks if a given sum H is achievable.
    For a fixed H, for each tooth i, the final upper tooth length U_i_final must satisfy:
    1. 0 <= U_i_final <= U_i  (Cannot increase length, must be non-negative)
    2. 0 <= D_i_final <= D_i  (Cannot increase length, must be non-negative)
    3. U_i_final + D_i_final = H (Condition 1 for fitting well)

    From (3), D_i_final = H - U_i_final. Substituting into (2):
    0 <= H - U_i_final <= D_i
    H - D_i <= U_i_final <= H

    Combining with (1):
    max(0, H - D_i) <= U_i_final <= min(U_i, H)

    Let [L_i(H), R_i(H)] be this initial possible range for U_i_final.
    L_i(H) = max(0, H - D[i])
    R_i(H) = min(U[i], H)

    We also have the constraint |U_i_final - U_{i+1}_final| <= X.
    This means U_{i+1}_final must be in [U_i_final - X, U_i_final + X].

    We can find the feasible range for each U_i_final by propagating constraints.
    Let [l_prime_i, r_prime_i] be the feasible range for U_i_final after considering teeth 1 to i.
    For i=1: [l_prime_1, r_prime_1] = [L_1(H), R_1(H)].
    For i > 1: U_i_final must be in [L_i(H), R_i(H)], AND it must satisfy |U_i_final - U_{i-1}_final| <= X
    for some U_{i-1}_final in [l_prime_{i-1}, r_prime_{i-1}].
    The values U_i_final that satisfy |U_i_final - U_{i-1}_final| <= X for *some* U_{i-1}_final in [l_prime_{i-1}, r_prime_{i-1}]
    are those in the range [min(u - X), max(u + X)] for u in [l_prime_{i-1}, r_prime_{i-1}],
    which is exactly [l_prime_{i-1} - X, r_prime_{i-1} + X].

    So, [l_prime_i, r_prime_i] = [L_i(H), R_i(H)] intersect [l_prime_{i-1} - X, r_prime_{i-1} + X].
    l_prime_i = max(L_i(H), l_prime_{i-1} - X)
    r_prime_i = min(R_i(H), r_prime_{i-1} + X)

    A sum H is achievable if and only if this feasible range [l_prime_i, r_prime_i] is non-empty (i.e., l_prime_i <= r_prime_i) for all i from 1 to N.
    """

    # Range for U_1 (0-indexed: U[0], D[0])
    l_current = max(0, H - D[0])
    r_current = min(U[0], H)

    # If the initial range is invalid (empty)
    if l_current > r_current:
        return False

    # Propagate constraints for U_2 to U_N
    for i in range(1, N):
        # Previous valid range for U_{i-1} was [l_current, r_current]
        l_prev = l_current
        r_prev = r_current

        # Initial range for U_i (0-indexed: U[i], D[i]) based on sum H
        # max(0, H - D[i]) <= U_i_final <= min(U[i], H)
        l_i = max(0, H - D[i])
        r_i = min(U[i], H)

        # Range for U_i_final based on U_{i-1}_final being in [l_prev, r_prev]
        # |U_i_final - U_{i-1}_final| <= X
        # U_{i-1}_final - X <= U_i_final <= U_{i-1}_final + X
        # To satisfy this for *some* U_{i-1}_final in [l_prev, r_prev],
        # U_i_final must be in [l_prev - X, r_prev + X]
        l_from_prev = l_prev - X
        r_from_prev = r_prev + X

        # The valid range for U_i_final is the intersection of its initial range and the range from the previous tooth
        l_current = max(l_i, l_from_prev)
        r_current = min(r_i, r_from_prev)

        # Check if the intersected range is valid (non-empty)
        if l_current > r_current:
            return False

    # If we successfully found a valid range for all U_i, then H is achievable
    return True

def solve():
    # Read N and X
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    X = int(line1[1])

    U = []
    D = []
    total_initial_sum = 0
    min_S_initial = float('inf') # Use float('inf') for initial min

    # Read U_i and D_i
    for _ in range(N):
        line = sys.stdin.readline().split()
        u = int(line[0])
        d = int(line[1])
        U.append(u)
        D.append(d)
        total_initial_sum += u + d
        min_S_initial = min(min_S_initial, u + d)

    # Binary search for the maximum possible H.
    # H must be an integer >= 0.
    # Also, U_i_final + D_i_final = H, and U_i_final <= U_i, D_i_final <= D_i.
    # Summing these inequalities: U_i_final + D_i_final <= U_i + D_i, so H <= U_i + D_i.
    # This must hold for all i, so H <= min(U_i + D_i) = min_S_initial.
    # Search range for H is [0, min_S_initial].
    # The check(H) function is monotonic: if check(H) is true, check(H') might be true or false for H' > H,
    # but check(H') is true for all H' in [0, H] is not guaranteed.
    # However, the minimum width of the valid interval R_i'(H) - L_i'(H) is a concave function of H.
    # We are looking for the maximum H where this minimum width >= 0.
    # This allows binary search for the maximum feasible H.
    low = 0
    high = min_S_initial
    optimal_H = 0 # Initialize with 0, as H=0 is always feasible

    while low <= high:
        mid = low + (high - low) // 2 # Integer division
        if check(mid, N, X, U, D):
            # mid is achievable, try larger H
            optimal_H = mid # mid is a candidate for the maximum H
            low = mid + 1
        else:
            # mid is not achievable, H must be smaller
            high = mid - 1

    # Minimum cost = Sum of initial lengths - Sum of final lengths
    # Sum of final lengths = Sum(U_i_final + D_i_final) = Sum(H) = N * optimal_H
    min_cost = total_initial_sum - N * optimal_H
    print(min_cost)

# Call the solve function
solve()