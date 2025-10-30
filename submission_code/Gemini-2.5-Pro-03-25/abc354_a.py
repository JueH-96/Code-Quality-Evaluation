import sys

def solve():
    # Read Takahashi's height H from standard input
    H = int(sys.stdin.readline())

    # Determine the height of the plant in the morning of day 'd'.
    # Day 0 morning: height = 0
    # Day 0 night: height increases by 2^0 = 1
    # Day 1 morning: height = 1
    # Day 1 night: height increases by 2^1 = 2
    # Day 2 morning: height = 1 + 2 = 3
    # Day 2 night: height increases by 2^2 = 4
    # Day 3 morning: height = 3 + 4 = 7
    # The height in the morning of day 'd', P(d), is the sum of growths from night 0 to night d-1:
    # P(d) = 2^0 + 2^1 + 2^2 + ... + 2^(d-1)
    # This is a geometric series sum, which equals 2^d - 1.

    # We need to find the first day 'd' such that the plant's height P(d) is strictly greater than H.
    # We need the smallest integer 'd' such that P(d) > H.
    # Substituting the formula for P(d):
    # 2^d - 1 > H
    # 2^d > H + 1

    # Let target = H + 1. We need the smallest integer 'd' such that 2^d > target.
    target = H + 1

    # The `int.bit_length()` method provides a way to find this 'd'.
    # For a positive integer `x`, `x.bit_length()` returns the smallest integer `k` such that `x < 2**k`.
    # Since H >= 1, target = H + 1 >= 2, so target is a positive integer.

    # Let k = target.bit_length(). By the definition of `bit_length()`, k is the smallest integer
    # such that target < 2**k.
    # We are looking for the smallest integer 'd' such that 2**d > target.

    # Let's test potential values for 'd':
    # 1. Try d = k: We know target < 2**k, which means 2**k > target. So, d = k satisfies the condition.
    # 2. Try d = k - 1: Since k is the *smallest* integer such that target < 2**k, it must be that
    #    target is NOT less than 2**(k-1). This implies target >= 2**(k-1).
    #    Therefore, 2**(k-1) is not strictly greater than target. So, d = k - 1 does not satisfy the condition 2**d > target.

    # Thus, the smallest integer 'd' that satisfies 2**d > target is k = target.bit_length().
    result_day = target.bit_length()

    # Print the result to standard output
    print(result_day)

# Execute the solve function
solve()