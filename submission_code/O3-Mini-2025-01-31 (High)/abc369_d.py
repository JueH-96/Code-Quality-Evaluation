def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))

    # Use a sufficiently small number as negative infinity.
    NEG_INF = -10**20

    # dp_even: maximum experience so far when the number of defeated monsters is even.
    # dp_odd: maximum experience so far when the number of defeated monsters is odd.
    dp_even = 0
    dp_odd = NEG_INF

    # Process each monster in order.
    for x in A:
        # Two decisions for each monster:
        # 1. Skip: the state does not change.
        # 2. Kill:
        #    - If prior state is even, the monster becomes an odd kill and adds x points.
        #    - If prior state is odd, the monster becomes an even kill and adds 2*x points.
        new_even = max(dp_even, dp_odd + 2 * x)  # New even state: either skip (remain dp_even) or kill from odd state.
        new_odd  = max(dp_odd,  dp_even + x)      # New odd state: either skip or kill from even state.
        dp_even, dp_odd = new_even, new_odd

    # The answer is the best achievable total experience,
    # regardless of whether the final defeat count is odd or even.
    ans = max(dp_even, dp_odd)
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()