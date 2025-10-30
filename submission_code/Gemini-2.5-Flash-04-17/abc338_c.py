import sys

# Read N
N = int(sys.stdin.readline())

# Read Q, A, B
Q = list(map(int, sys.stdin.readline().split()))
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

# Calculate the maximum possible number of servings of dish A
# if we were to make 0 servings of dish B.
# Constraint: a * A[i] <= Q[i] for all i.
# If A[i] > 0, then a <= Q[i] // A[i].
# The maximum a is the minimum of Q[i] // A[i] over all i where A[i] > 0.
max_a_val = sys.maxsize
# The problem statement guarantees there is at least one i such that A_i >= 1,
# so this loop will find at least one such i and max_a_val will become finite.
for i in range(N):
    if A[i] > 0:
        max_a_val = min(max_a_val, Q[i] // A[i])

# max_a_val is finite and represents the maximum possible servings of A
# we can make using only the ingredients, assuming we make 0 servings of B.
# Any number of servings of A greater than max_a_val is impossible
# even without making any servings of B.
# So, we only need to iterate through the possible number of servings of A
# from 0 up to max_a_val.

max_total_servings = 0

# Iterate through all possible number of servings of dish A
# from 0 up to max_a_val (inclusive).
for a in range(max_a_val + 1):
    # For a given number of servings of dish A ('a'), calculate the maximum
    # number of servings of dish B ('b') that can also be made.
    # The quantity of ingredient i remaining after making 'a' servings of A is:
    # rem_Q_i = Q[i] - a * A[i].
    # Since we iterate 'a' only up to max_a_val, we know that for any i where A[i]>0,
    # a <= Q[i] // A[i], which implies a * A[i] <= Q[i].
    # If A[i] == 0, then a * A[i] = 0, and Q[i] >= 1, so 0 <= Q[i] is always true.
    # Thus, for any 'a' in this range, rem_Q_i = Q[i] - a * A[i] is guaranteed to be >= 0 for all i.

    # The constraint for dish B is b * B[i] <= rem_Q_i for all i.
    # If B[i] > 0, then b <= rem_Q_i // B[i].
    # The maximum possible integer b is the minimum of rem_Q_i // B[i]
    # over all i where B[i] > 0.
    current_max_b = sys.maxsize # Use a large value to find the minimum.

    # The problem statement guarantees there is at least one i such that B_i >= 1,
    # so there will be at least one ingredient constraint that limits b.
    # current_max_b will be updated to a finite value.
    for i in range(N):
        # Remaining quantity for ingredient i
        rem_Q_i = Q[i] - a * A[i]

        if B[i] > 0:
            b_limit = rem_Q_i // B[i]
            current_max_b = min(current_max_b, b_limit)
        # If B[i] == 0, this ingredient does not restrict 'b', provided that
        # a * A[i] <= Q[i], which we already know is true for a <= max_a_val.

    # After checking all ingredients, current_max_b holds the maximum number of
    # servings of dish B possible for this specific number of servings of dish A ('a').
    # Since rem_Q_i >= 0 and we only take minimum over i where B[i] > 0,
    # and there is at least one such i, current_max_b will be a non-negative integer.

    # The total number of servings for this pair (a, current_max_b) is a + current_max_b.
    # We update the overall maximum total servings found so far.
    max_total_servings = max(max_total_servings, a + current_max_b)

# After iterating through all possible values of 'a', max_total_servings
# holds the overall maximum total number of servings.
print(max_total_servings)