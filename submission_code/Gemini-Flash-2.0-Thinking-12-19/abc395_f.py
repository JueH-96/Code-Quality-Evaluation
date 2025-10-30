import sys

# Function to check if a given H is feasible
def is_feasible(H, N, X, U_orig, D_orig):
    # Feasible range for U_i given H is [max(0, H - D_orig[i]), min(U_orig[i], H)]
    # Let L_i(H) = max(0, H - D_orig[i]) and R_i(H) = min(U_orig[i], H)

    # Calculate forward feasible range [l_i, r_i]
    # l[i] is the minimum possible value of U_i considering constraints from 0 to i
    # r[i] is the maximum possible value of U_i considering constraints from 0 to i
    l = [0] * N
    r = [0] * N

    L_0 = max(0, H - D_orig[0])
    R_0 = min(U_orig[0], H)
    
    l[0] = L_0
    r[0] = R_0

    if l[0] > r[0]:
        return False

    for i in range(1, N):
        L_i = max(0, H - D_orig[i])
        R_i = min(U_orig[i], H)
        
        l[i] = max(L_i, l[i-1] - X)
        r[i] = min(R_i, r[i-1] + X)

        if l[i] > r[i]:
            return False

    # Calculate backward feasible range [l'_i, r'_i]
    # l_prime[i] is the minimum possible value of U_i considering constraints from i to N-1
    # r_prime[i] is the maximum possible value of U_i considering constraints from i to N-1
    l_prime = [0] * N
    r_prime = [0] * N

    L_N_1 = max(0, H - D_orig[N-1])
    R_N_1 = min(U_orig[N-1], H)

    l_prime[N-1] = L_N_1
    r_prime[N-1] = R_N_1

    if l_prime[N-1] > r_prime[N-1]:
        return False

    for i in range(N - 2, -1, -1):
        L_i = max(0, H - D_orig[i])
        R_i = min(U_orig[i], H)

        l_prime[i] = max(L_i, l_prime[i+1] - X)
        r_prime[i] = min(R_i, r_prime[i+1] + X)

        if l_prime[i] > r_prime[i]:
            return False

    # Check if the intersection of forward and backward feasible ranges is non-empty for all i
    # A feasible sequence U_0, ..., U_{N-1} exists iff for all i, max(l[i], l_prime[i]) <= min(r[i], r_prime[i])
    for i in range(N):
        if max(l[i], l_prime[i]) > min(r[i], r_prime[i]):
            return False

    return True

def solve():
    N, X = map(int, sys.stdin.readline().split())
    U_orig = [0] * N
    D_orig = [0] * N
    initial_total_sum = 0
    H_upper = float('inf') # Use a large number initially

    for i in range(N):
        U_orig[i], D_orig[i] = map(int, sys.stdin.readline().split())
        initial_total_sum += U_orig[i] + D_orig[i]
        H_upper = min(H_upper, U_orig[i] + D_orig[i])

    # Binary search for the maximum feasible H
    # Search space for H is [0, H_upper] inclusive.
    # Use [low, high) range for binary search: [0, H_upper + 1)
    low = 0
    high = H_upper + 1 # Exclusive upper bound

    # Find the largest H in [0, H_upper] such that is_feasible(H) is True.
    # The property is monotonic decreasing (non-increasing): if is_feasible(H) is True, then is_feasible(H') is True for any 0 <= H' <= H.
    # Binary search finds the boundary.
    # Template: low = L, high = R + 1. while high - low > 1: mid = low + (high - low) // 2. If P(mid): low = mid. Else: high = mid. Answer is low.
    # Here P(mid) is is_feasible(mid).

    max_H = 0 # H=0 is always feasible (U_i=0, D_i=0). The max feasible H will be at least 0.

    while high - low > 1:
        mid = low + (high - low) // 2
        if is_feasible(mid, N, X, U_orig, D_orig):
            # mid is feasible, we can achieve this sum H. Try higher.
            # The maximum feasible H could be mid or larger.
            low = mid
        else:
            # mid is not feasible. H must be less than mid.
            # The maximum feasible H must be less than mid.
            high = mid

    # After the loop terminates, low is the largest value for which is_feasible(low) is True,
    # and high = low + 1 is the smallest value for which is_feasible(high) is False.
    max_H = low

    # Minimum cost = initial_total_sum - N * max_H
    min_cost = initial_total_sum - N * max_H

    print(min_cost)

solve()