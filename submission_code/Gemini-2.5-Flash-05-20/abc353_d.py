import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    total_sum = 0
    prefix_sum_A = 0 # Stores sum of A[0]...A[j-1] modulo MOD

    # Iterate through the array A using index `j`.
    # For each A[j], we consider all pairs (A[k], A[j]) where k < j.
    # The sum of f(A[k], A[j]) for k from 0 to j-1 is:
    #   sum_{k=0}^{j-1} (A[k] * 10^digits(A[j]) + A[j])
    # = (sum_{k=0}^{j-1} A[k]) * 10^digits(A[j]) + (sum_{k=0}^{j-1} A[j])
    # = (prefix_sum_A) * 10^digits(A[j]) + j * A[j]

    for j in range(N):
        current_A_val = A[j]

        # Calculate number of digits for current_A_val (A[j])
        # This is needed for the 10^digits(A[j]) term.
        num_digits_current_A = len(str(current_A_val))
        power_of_10_for_current_A = pow(10, num_digits_current_A, MOD)

        # Contribution of current_A_val (A[j]) itself from the 'second part' (A_j)
        # For each k from 0 to j-1, A[j] is added once. There are 'j' such k's.
        # So, this part contributes `j * A[j]`.
        total_sum = (total_sum + j * current_A_val) % MOD

        # Contribution from the sum of previous elements (A[0]...A[j-1])
        # multiplied by 10^digits(A[j]).
        # This sum is `(sum_{k=0}^{j-1} A[k]) * 10^digits(A[j])`.
        # `prefix_sum_A` currently holds `sum_{k=0}^{j-1} A[k]`.
        total_sum = (total_sum + prefix_sum_A * power_of_10_for_current_A) % MOD

        # Update prefix_sum_A by adding current_A_val (A[j])
        # This updated `prefix_sum_A` will be used in the next iteration for A[j+1]
        # (it will contain sum A[0]...A[j]).
        prefix_sum_A = (prefix_sum_A + current_A_val) % MOD

    sys.stdout.write(str(total_sum) + "
")

solve()