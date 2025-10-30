import sys

# Read N
N = int(sys.stdin.readline())

# Read S_i values and populate isPresent array.
# The maximum possible value for S_i is 10^6.
MAX_VAL = 1000000
# Using a list as a boolean array. Indices 1 to MAX_VAL will be used.
# Size MAX_VAL + 1 because values are up to MAX_VAL.
isPresent = [False] * (MAX_VAL + 1)

# Reading S values. We don't need to store them in a list for this algorithm.
# Just populate the isPresent array.
for _ in range(N):
    s_val = int(sys.stdin.readline())
    # The constraints guarantee 1 <= s_val <= 10^6
    # We only care about values within the possible range [1, MAX_VAL]
    if 1 <= s_val <= MAX_VAL:
        isPresent[s_val] = True

count = 0

# Iterate through possible values for B (the middle element of the triplet)
# B can range from 1 up to MAX_VAL. We only proceed if B is present in S.
# The boolean array allows us to check presence efficiently.
for B_val in range(1, MAX_VAL + 1):
    if isPresent[B_val]:
        # If B_val is in S, iterate through possible positive common differences k.
        # A = B_val - k, C = B_val + k.
        # We require A >= 1 and C <= MAX_VAL.
        # B_val - k >= 1  => k <= B_val - 1
        # B_val + k <= MAX_VAL => k <= MAX_VAL - B_val
        # Also k must be positive, so 1 <= k.
        # Thus, 1 <= k <= min(B_val - 1, MAX_VAL - B_val).

        # The upper bound for k can be pre-calculated.
        # If B_val = 1, min(0, ...) = 0. If B_val = MAX_VAL, min(..., 0) = 0.
        # If B_val is close to 1 or MAX_VAL, max_k is small.
        # If B_val is close to MAX_VAL/2, max_k is close to MAX_VAL/2.
        max_k = min(B_val - 1, MAX_VAL - B_val)

        # Iterate through k from 1 up to max_k.
        # If max_k < 1, range(1, max_k + 1) will be empty, which is correct.
        for k in range(1, max_k + 1):
            A_val = B_val - k
            C_val = B_val + k

            # Check if A_val and C_val are present in S.
            # Due to the range of k [1, max_k], A_val is in [1, B_val - 1]
            # and C_val is in [B_val + 1, MAX_VAL].
            # So A_val >= 1 and C_val <= MAX_VAL are guaranteed.
            # A_val < B_val < C_val is also guaranteed since k >= 1.
            # Thus, A, B, C are distinct elements forming an arithmetic progression.
            # We only need to check if A_val and C_val are in S.
            # The indices A_val and C_val are guaranteed to be within [1, MAX_VAL]
            # by the loop limits for k, so no bounds check needed for isPresent array access.
            if isPresent[A_val] and isPresent[C_val]:
                count += 1

# Print the total count of fine triplets.
print(count)