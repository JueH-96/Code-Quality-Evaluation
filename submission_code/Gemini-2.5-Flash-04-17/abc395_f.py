# YOUR CODE HERE
import sys

# The feasibility check function
def is_feasible(H, N, X, U_orig, D_orig):
    """
    Checks if a given sum H (U_i + D_i = H for all i) is feasible,
    considering original tooth lengths and the difference constraint X.
    """
    # The required range for U_i is [max(0, H - D_orig[i]), min(U_orig[i], H)].
    # Let's track the possible range [current_l, current_r] for U_i.

    # Initial range for U_1 (index 0)
    current_l = max(0, H - D_orig[0])
    current_r = min(U_orig[0], H)

    # If the initial range for U_1 is empty, H is not feasible.
    # This check is technically redundant if H <= min_s_orig, but good practice.
    if current_l > current_r:
        return False

    # Propagate constraints for U_2 to U_N (index 1 to N-1)
    # The constraint is |U_i - U_{i+1}| <= X for 1 <= i < N (1-based index).
    # In 0-based index, this is |U[i] - U[i+1]| <= X for 0 <= i <= N-2.
    # When calculating the range for U[i] (index i), the constraint involving the previous tooth U[i-1] (index i-1) is |U[i-1] - U[i]| <= X.
    # So U[i] must be in the range [U[i-1] - X, U[i-1] + X].
    # If U[i-1] is in [current_l_prev, current_r_prev], then U[i] must be in [current_l_prev - X, current_r_prev + X].
    # The loop below calculates the feasible range for U_i (index i) based on the feasible range of U_{i-1} (index i-1).
    # The loop variable `i` will represent the 0-based index of the tooth being processed (from 1 to N-1).
    for i in range(1, N):
        # Bounds for U_i (index i) based on original lengths and H
        L_i = max(0, H - D_orig[i])
        R_i = min(U_orig[i], H)

        # Bounds for U_i (index i) based on U_{i-1}'s (index i-1) feasible range [current_l, current_r]
        # and the difference constraint |U_i - U_{i-1}| <= X
        # U_{i-1} - X <= U_i <= U_{i-1} + X
        # This means U_i must be in the range [current_l - X, current_r + X]

        next_l_from_prev = current_l - X
        next_r_from_prev = current_r + X

        # The feasible range for U_i (index i) is the intersection of [L_i, R_i] and [next_l_from_prev, next_r_from_prev]
        current_l = max(L_i, next_l_from_prev)
        current_r = min(R_i, next_r_from_prev)

        # If the feasible range for U_i becomes empty, H is not feasible
        if current_l > current_r:
            return False

    # If we successfully found a non-empty feasible range for U_N,
    # then there exists a sequence U_1, ..., U_N satisfying all constraints for this H.
    return True


# The main solver function
def solve():
    # Read N and X
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    X = int(line1[1])

    U_orig = [0] * N
    D_orig = [0] * N
    total_initial_sum = 0
    min_s_orig = -1 # Placeholder, will be set with the first tooth pair sum

    # Read tooth lengths and calculate total initial sum and minimum original sum
    for i in range(N):
        u, d = map(int, sys.stdin.readline().split())
        U_orig[i] = u
        D_orig[i] = d
        current_s_orig = u + d
        total_initial_sum += current_s_orig
        
        if i == 0:
            min_s_orig = current_s_orig
        else:
            min_s_orig = min(min_s_orig, current_s_orig)

    # Binary search for the maximum feasible H.
    # The theoretical maximum value for H is the minimum of the original sums (U_i + D_i).
    # The theoretical minimum value for H is 0.
    low = 0
    high = min_s_orig
    max_feasible_h = 0 # H=0 is always feasible (by grinding all teeth to 0), so initialize max_feasible_h to 0

    while low <= high:
        mid = low + (high - low) // 2 # Calculate midpoint to avoid potential overflow with (low + high) // 2

        if is_feasible(mid, N, X, U_orig, D_orig):
            # If mid is a feasible value for H, it means we might be able to achieve
            # this sum H. We store it as a potential maximum and try larger H values.
            max_feasible_h = mid
            low = mid + 1
        else:
            # If mid is not feasible, it means H is too high. We need to try smaller H values.
            high = mid - 1

    # The minimum cost to make teeth fit well with the maximum possible H_max is:
    # Cost = (Sum of original lengths) - (Sum of final lengths)
    # Sum of original lengths = total_initial_sum
    # Sum of final lengths = sum(U_i + D_i) for the optimal configuration
    # Since U_i + D_i = H_max for all i in the optimal configuration,
    # Sum of final lengths = N * H_max
    min_cost = total_initial_sum - N * max_feasible_h

    # Print the minimum cost
    print(min_cost)

# Execute the solver function if the script is run directly
if __name__ == "__main__":
    solve()