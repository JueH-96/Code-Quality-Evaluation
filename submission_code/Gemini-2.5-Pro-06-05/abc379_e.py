import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        N_str = input()
        if not N_str: return
        N = int(N_str)
        S = input().strip()
    except (IOError, ValueError):
        return

    # Convert string of digits to a list of integers
    s_digits = [int(d) for d in S]

    # The total sum can be expressed as sum_{p=0 to N-1} Coeff_p * 10^p.
    # The coefficient of 10^p, Coeff_p, is sum_{k=0 to N-1-p} (k+1) * s_k.

    # We can precompute prefix sums P_i = sum_{k=0 to i} (k+1) * s_k.
    # Then Coeff_p = P_{N-1-p}.
    P = [0] * N
    current_P_sum = 0
    for i in range(N):
        current_P_sum += (i + 1) * s_digits[i]
        P[i] = current_P_sum

    # Coeff_p for p=0..N-1 is P_{N-1-p}.
    # Let's call the list of coefficients C. C[p] = Coeff_p.
    # This means C is the reversed P array.
    C = P[::-1]

    # Now, compute the final number from its coefficients: sum_{p=0 to N-1} C[p] * 10^p.
    # This is done by simulating manual addition with carries, digit by digit.
    result_digits = []
    carry = 0

    # Loop through coefficients from C_0 upwards (least significant power of 10)
    for i in range(N):
        val = C[i] + carry
        result_digits.append(str(val % 10))
        carry = val // 10

    # Process any remaining carry
    while carry > 0:
        result_digits.append(str(carry % 10))
        carry //= 10

    # The digits were generated from right to left, so reverse to get the final number.
    print("".join(result_digits[::-1]))

solve()