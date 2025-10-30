def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1]
    costs = list(map(int, input_data[2:]))

    # Convert S into a list of integer bits (0 or 1)
    sbits = [int(ch) for ch in S]

    # cost0[i] = cost to make position i (1-based) match the "alt0" pattern:
    #   alt0:  1 -> 0, 2 -> 1, 3 -> 0, 4 -> 1, ...
    # cost1[i] = cost to make position i (1-based) match the "alt1" pattern:
    #   alt1:  1 -> 1, 2 -> 0, 3 -> 1, 4 -> 0, ...
    # We'll store these in 0-based arrays for convenience.
    cost0 = [0]*N
    cost1 = [0]*N
    for i in range(N):
        # i+1 is the 1-based index
        want0 = (i % 2)  # alt0: odd->0, even->1 => if i is even(0-based), i+1 is odd => want0=0
        # Actually easier way: alt0[1]=0 => i=0 => want0=0, i=1 => want0=1, i=2 => want0=0...
        # So alt0[i+1] = (i+1-1) % 2 = i % 2, indeed.
        # But we want "0,1,0,1..." for i=0,1,2,3 => i%2 => 0,1,0,1 is correct.
        # So if i%2=0 => alt0 is 0, if i%2=1 => alt0 is 1
        bit_s = sbits[i]
        if bit_s != (i % 2):  # alt0 bit at position i => i%2
            cost0[i] = costs[i]
        # alt1 is the opposite of alt0
        if bit_s != (1 - (i % 2)):
            cost1[i] = costs[i]

    # Build prefix sums for cost0 and cost1 to quickly get the cost of
    # making the prefix [1..k] follow alt0 or alt1 exactly (fully alternating).
    p0 = [0]*(N+1)
    p1 = [0]*(N+1)
    for i in range(1, N+1):
        p0[i] = p0[i-1] + cost0[i-1]
        p1[i] = p1[i-1] + cost1[i-1]

    # Now build suffixCost[b][i]: the cost to transform S[i..N] (1-based) into an
    # alternating pattern starting with bit b at index i.
    # We'll store in two 1D arrays of length N+2 for b=0 or b=1:
    # suffixCost0[i] = cost if T[i] = 0, suffixCost1[i] = cost if T[i] = 1
    suffixCost0 = [0]*(N+2)
    suffixCost1 = [0]*(N+2)

    # Fill from the back
    for i in range(N, 0, -1):
        # cost to make S[i] (1-based) into 0:
        c0 = costs[i-1] if sbits[i-1] != 0 else 0
        c1 = costs[i-1] if sbits[i-1] != 1 else 0
        suffixCost0[i] = c0 + suffixCost1[i+1]  # T[i] = 0 => next i+1 = 1
        suffixCost1[i] = c1 + suffixCost0[i+1]  # T[i] = 1 => next i+1 = 0

    # We want exactly one break at position k (1 <= k <= N-1).
    # We also choose a in {0,1} for T[1].
    # Then T[1..k] is a fully alternating prefix from bit a,
    # T[k+1] = T[k] (the break),
    # T[k+2..N] is an alternating suffix starting from 1 - T[k+1].
    # T[k] = a ^ ((k-1) mod 2) if we are following a-based alternation in the prefix.
    # We'll compute the cost for each (a, k) and take the minimum.

    INF = 10**20
    ans = INF

    for a in (0, 1):
        # prefix cost array = p0 if a=0, p1 if a=1
        pref = p0 if a == 0 else p1
        for k in range(1, N):  # k is the break position
            # bit at position k in the prefix:
            # alt(a) for index k => a ^ ((k-1) mod 2) if we think 1-based indexing for T.
            # But since a is the bit at T[1], T[2] = 1-a, etc...
            # In simpler form: b = a ^ ((k-1) % 2)
            b = a ^ ((k-1) & 1)
            # cost of prefix up to k:
            cost_prefix = pref[k]  # sum cost to match alt(a) from 1..k
            # cost to make T[k+1] = b
            # S[k] (0-based) is the (k+1)-th character in 1-based indexing
            flip_kp1 = costs[k] if sbits[k] != b else 0
            # now T[k+2] must be 1-b, T[k+3] = b, etc => suffixCost[1-b][k+2]
            # watch out if k+2 > N => suffix cost is 0
            if k+2 <= N:
                cost_suffix = suffixCost1[b][k+2]  # but we need the correct b array
                # Actually we have suffixCost0[i] if T[i]=0, suffixCost1[i] if T[i]=1
                # We want T[k+2] = 1-b. So the cost is suffixCost(1-b, k+2).
                if (1-b) == 0:
                    cost_suffix = suffixCost0[k+2]
                else:
                    cost_suffix = suffixCost1[k+2]
            else:
                cost_suffix = 0

            total_cost = cost_prefix + flip_kp1 + cost_suffix
            if total_cost < ans:
                ans = total_cost

    # However, there's one subtlety: The above construction always imposes that
    # T[1..k] is strictly alternating (no break). That is fine, but it automatically
    # ensures exactly one break at k. This covers all possible "exactly 1 break"
    # strings of length N, so we haven't missed any. Thus ans is correct.

    print(ans)
    
# Call main() to ensure we get a result.
main()