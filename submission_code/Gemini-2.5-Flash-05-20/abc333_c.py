import sys

def solve():
    N = int(sys.stdin.readline())

    # Determine the maximum length (number of digits) of repunit needed.
    #
    # We are looking for the N-th smallest sum of three repunits.
    # The maximum value of N is 333.
    #
    # Let's consider the possible range of repunits.
    # R(1) = 1
    # R(2) = 11
    # R(3) = 111
    # ...
    #
    # The smallest sum is 1+1+1 = 3.
    # The sample case for N=333 gives the output 112222222233.
    # We can decompose this number:
    # 112222222233 = 111111111111 (R(12)) + 1111111111 (R(10)) + 11 (R(2)).
    # This decomposition shows that repunits up to R(12) (a repunit with 12 digits)
    # might be involved in the sums.
    #
    # The number of unique sums formed by choosing 3 repunits (with replacement)
    # from a set of `k` repunits (R(1) to R(k)) is given by the combinations with repetition formula:
    # C(k + 3 - 1, 3) = C(k + 2, 3) = (k+2)(k+1)k / 6.
    # For k = 12: C(12 + 2, 3) = C(14, 3) = (14 * 13 * 12) / (3 * 2 * 1) = 14 * 13 * 2 = 364.
    # Since N <= 333, and 364 >= 333, generating sums using repunits up to R(12)
    # will provide enough unique sums to find the N-th one.
    # Therefore, max_k = 12 is sufficient.
    max_k = 12

    # Step 1: Generate all necessary repunits
    repunits = []
    current_rep = 0
    for _ in range(max_k): # This loop generates R(1) through R(max_k)
        current_rep = current_rep * 10 + 1
        repunits.append(current_rep)
    # After this loop, repunits will contain [1, 11, 111, ..., R(12)]

    # Step 2: Generate all possible sums of three repunits
    sums = set() # Use a set to automatically handle unique sums
    # Iterate through all combinations of three repunits.
    # The order of r1, r2, r3 does not affect their sum, and the set handles uniqueness.
    # This triple loop performs max_k^3 iterations (12^3 = 1728), which is very fast.
    for r1 in repunits:
        for r2 in repunits:
            for r3 in repunits:
                sums.add(r1 + r2 + r3)

    # Step 3: Convert the set of sums to a list and sort it
    sorted_sums = sorted(list(sums))

    # Step 4: Print the N-th smallest sum
    # In a 0-indexed list, the N-th element is at index N-1.
    sys.stdout.write(str(sorted_sums[N-1]) + "
")

solve()