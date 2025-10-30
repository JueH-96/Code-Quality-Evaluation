def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1]
    C = list(map(int, input_data[2:]))

    # Idea:
    # We want T to be a "good" string, i.e. it has exactly one pair of consecutive equal bits.
    # Another way of saying that is:
    #  - T must be alternating everywhere, except for exactly one index i in [1..N-1]
    #    where T[i] = T[i+1].
    #
    # We can think of T as one of two "base" alternating patterns:
    #   alt0: 0 1 0 1 0 1 ...
    #   alt1: 1 0 1 0 1 0 ...
    #
    # except that at exactly one place (say i), we "force" T[i] = T[i+1], 
    # then continue alternating after that.  We wish to achieve this with minimum flipping cost.
    #
    # We will use the following construction:
    #  1) Precompute prefix-costs to match alt0 and alt1 up to index i.
    #  2) Precompute suffix-costs to match an alternating pattern *from* index i to the end,
    #     but crucially that suffix pattern can start with '0' or '1' independently of i's position.
    #  3) For each i in [1..N-1], we consider two cases:
    #       (a) The prefix (1..i) matches alt0 (or alt1),
    #           T[i+1] is forced to be T[i] from that prefix,
    #           then from i+2..N we alternate starting with the opposite bit of T[i+1].
    #     We compute the cost of that arrangement, and take the minimum over all i and both base patterns alt0/alt1.
    #
    # This yields an O(N) solution.
    #
    # Implementation details:
    #  - Let alt0[i] = '0' if i is odd, '1' if i is even (1-based index).
    #  - Let alt1[i] = '1' if i is odd, '0' if i is even.
    #
    #  - prefix0[i] = total cost to flip S[1..i] into the first i bits of alt0.
    #  - prefix1[i] = total cost to flip S[1..i] into the first i bits of alt1.
    #
    #  - We also need suffix arrays that let us quickly compute the cost to flip S[j..N]
    #    into an alternating string that starts with '0' or starts with '1'.
    #    We'll denote them as suffix0[j], suffix1[j]:
    #      suffix0[j] = cost to flip S[j..N] to an alternating pattern starting with '0',
    #      suffix1[j] = cost to flip S[j..N] to an alternating pattern starting with '1'.
    #
    # Then for a break at position i (1 <= i < N):
    #   - Suppose we choose the prefix to match alt0 up to i; cost = prefix0[i].
    #   - We force T[i+1] = T[i] from that prefix. If T[i] = alt0[i], call that bit c0 (which is '0' or '1').
    #   - Flipping cost at i+1 if needed: extra_cost = (S[i+1] != c0) ? C[i+1] : 0.
    #   - Then from i+2..N, we want to alternate starting from bit (1 - c0), i.e. if c0='0' => next bit='1', etc.
    #     So if c0='0', we want suffix1[i+2], else suffix0[i+2].  (Handle i+2 out of range carefully.)
    #   - Summation gives total cost for that scenario.
    # We do similarly if the prefix is alt1 up to i, then T[i+1] = alt1[i], etc.
    # Take the minimum over all i and the two base patterns.
    #
    # Edge case: i+2 could be > N, in which case suffixX[i+2] = 0 by definition (empty substring cost).
    #
    # Let's implement.

    # For convenience, make S and C be 1-indexed in the code:
    S = "_" + S  # so that S[i] corresponds to i in [1..N]
    C = [0] + C  # likewise

    # Precompute prefix costs for alt0 and alt1.
    # alt0[i] = '0' if i is odd, '1' if i is even
    # alt1[i] = '1' if i is odd, '0' if i is even
    prefix0 = [0] * (N + 1)
    prefix1 = [0] * (N + 1)
    for i in range(1, N + 1):
        # cost if we want alt0 for position i
        want0 = '0' if (i % 2 == 1) else '1'
        prefix0[i] = prefix0[i - 1] + (C[i] if S[i] != want0 else 0)

        # cost if we want alt1 for position i
        want1 = '1' if (i % 2 == 1) else '0'
        prefix1[i] = prefix1[i - 1] + (C[i] if S[i] != want1 else 0)

    # Precompute suffix costs for an alternating pattern "starting with 0" or "starting with 1"
    # We'll define suffix0[i] = cost to flip S[i..N] so that T[i] = '0', T[i+1] = '1', T[i+2]='0', ...
    # Similarly suffix1[i].
    #
    # A neat way is to use a DP:
    #   suffix0[i] = (S[i] != '0' ? C[i] : 0) + suffix1[i+1]
    #   suffix1[i] = (S[i] != '1' ? C[i] : 0) + suffix0[i+1]
    # with suffixX[N+1] = 0 as base.
    suffix0 = [0] * (N + 2)
    suffix1 = [0] * (N + 2)
    for i in range(N, 0, -1):
        suffix0[i] = suffix1[i + 1] + (C[i] if S[i] != '0' else 0)
        suffix1[i] = suffix0[i + 1] + (C[i] if S[i] != '1' else 0)

    # Now try all breaks i in [1..N-1].
    # cost0_i = prefix0[i] + flipCostAt(i+1) + suffixX[i+2],
    #    where flipCostAt(i+1) ensures T[i+1] = alt0[i],
    #    alt0[i] = '0' if i odd else '1'.
    #    if alt0[i] = '0', we want suffix1[i+2], else suffix0[i+2].  (because next bit after T[i+1] must be 1 - T[i+1])
    #
    # cost1_i is analogous for alt1 prefix.
    #
    # We take the minimum among all i and both patterns.

    ans = float('inf')
    for i in range(1, N):
        # alt0[i]
        prefix_cost_0 = prefix0[i]  # cost to match alt0 for first i bits
        c0 = '0' if (i % 2 == 1) else '1'  # the bit alt0[i]
        flip_cost_0 = C[i + 1] if S[i + 1] != c0 else 0
        if c0 == '0':
            # Next suffix = suffix1 from i+2
            sf = suffix1[i + 2] if (i + 2 <= N) else 0
        else:
            # c0 == '1'
            sf = suffix0[i + 2] if (i + 2 <= N) else 0
        cost0_i = prefix_cost_0 + flip_cost_0 + sf
        if cost0_i < ans:
            ans = cost0_i

        # alt1[i]
        prefix_cost_1 = prefix1[i]
        c1 = '1' if (i % 2 == 1) else '0'  # the bit alt1[i]
        flip_cost_1 = C[i + 1] if S[i + 1] != c1 else 0
        if c1 == '0':
            sf = suffix1[i + 2] if (i + 2 <= N) else 0
        else:
            sf = suffix0[i + 2] if (i + 2 <= N) else 0
        cost1_i = prefix_cost_1 + flip_cost_1 + sf
        if cost1_i < ans:
            ans = cost1_i

    print(ans)

# Don't forget to call main()
if __name__ == "__main__":
    main()