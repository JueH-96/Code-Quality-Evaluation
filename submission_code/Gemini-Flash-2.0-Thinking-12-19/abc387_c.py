import sys
# Increase recursion depth for DP
sys.setrecursionlimit(2000) # Max length is 19, DP depth is length + 1

memo_len_D = {}
S_str_len_D = ""
D_len_D = 0

def count_len_D_snake_dp(index, tight, msd_val, has_violated):
    """
    Counts numbers of length D_len_D that are <= int(S_str_len_D)
     and satisfy the Snake digit condition (MSD > other digits).
    The first digit is implicitly handled as non-zero by the initial call structure.
    msd_val stores the value of the first digit (MSD).
    """
    state = (index, tight, msd_val, has_violated)
    if state in memo_len_D:
        return memo_len_D[state]

    # Base case: Successfully placed D_len_D digits.
    # A number formed by this DP path is of length D_len_D (first digit non-zero).
    # msd_val was set at index 0 and is guaranteed > 0.
    # We just need to check if any subsequent digit violated the condition.
    if index == D_len_D:
        return 1 if not has_violated else 0

    limit = int(S_str_len_D[index]) if tight else 9
    ans = 0

    # Determine the range of digits we can place at the current index.
    # The first digit (index == 0) must be non-zero (1 to 9).
    # Subsequent digits (index > 0) can be 0 to 9.
    start_digit = 1 if index == 0 else 0

    for digit in range(start_digit, limit + 1):
        new_tight = tight and (digit == limit)
        new_has_violated = has_violated
        current_msd = msd_val # msd_val was set at index 0

        if index == 0:
            # This is the first digit (MSD). Store its value.
            current_msd = digit
            # Initially, no digits have been placed *after* this MSD, so violated is False.
            new_has_violated = False

        else:
            # Not the first digit (index > 0). Check violation against the stored MSD.
            # msd_val is the value of the first digit (which was > 0 because index 0 loop starts from 1).
            if digit >= msd_val:
                new_has_violated = True
            # msd_val remains the same.

        # Recursively call for the next index.
        ans += count_len_D_snake_dp(index + 1, new_tight, current_msd, new_has_violated)

    memo_len_D[state] = ans
    return ans


def calculate_count(N):
    """
    Counts Snake numbers X such that 10 <= X <= N.
    """
    # Snake numbers must be >= 10.
    if N < 10:
        return 0

    S_str_local = str(N)
    D_local = len(S_str_local)
    total_snake_count = 0

    # 1. Count Snake numbers with length k, where 2 <= k < D_local.
    # These numbers are always less than any number of length D_local.
    # A Snake number of length k has a non-zero first digit (MSD) and k-1 other digits.
    # The smallest length for a Snake number is 2.
    for k in range(2, D_local):
        # For a fixed length k, iterate through possible MSD m (1 to 9).
        # The first digit is m.
        # The remaining k-1 digits must all be strictly less than m.
        # Each of the k-1 digits can be any value from 0 to m-1. There are m choices.
        # Number of such numbers with MSD m and length k is 1 * (m)^(k-1).
        for m in range(1, 10):
             total_snake_count += pow(m, k - 1)

    # 2. Count Snake numbers with length D_local that are <= N.
    # These are numbers X = d_0 d_1 ... d_{D_local-1} such that int(X) <= N,
    # d_0 > 0, and d_0 > d_i for all i from 1 to D_local-1.
    # This requires digit DP because of the upper bound N.
    global memo_len_D, S_str_len_D, D_len_D
    S_str_len_D = S_str_local
    D_len_D = D_local
    memo_len_D = {} # Clear memoization table for the new N

    # Initial DP call for length D_local numbers.
    # index=0 (first digit), tight=True (restricted by S_str_len_D),
    # msd_val=0 (will be set by the loop at index 0), has_violated=False.
    # The DP starts the first digit loop from 1, ensuring non-zero first digit.
    total_snake_count += count_len_D_snake_dp(0, True, 0, False)

    return total_snake_count

# Read input from stdin
L, R = map(int, sys.stdin.readline().split())

# The number of Snake numbers in [L, R] is count(R) - count(L-1).
# calculate_count(N) counts Snake numbers <= N (and >= 10).
# calculate_count(L-1) counts Snake numbers <= L-1 (and >= 10).
# Subtracting gives the count in [L, R].
# Since L >= 10, L-1 might be < 10 (if L=10).
# calculate_count(N) correctly returns 0 for N < 10, so calculate_count(L-1) works correctly even if L-1 < 10.
result = calculate_count(R) - calculate_count(L - 1)

# Print output to stdout
print(result)