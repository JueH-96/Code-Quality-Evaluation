def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    mod = 998244353
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    maxA = 100000

    # Precompute Euler's totient function: phi[1..maxA]
    phi = list(range(maxA+1))
    for i in range(2, maxA+1):
        if phi[i] == i:  # i is prime
            for j in range(i, maxA+1, i):
                phi[j] -= phi[j] // i

    # Precompute divisors for each integer from 1 to maxA.
    # divisors[x] will be a list of all divisors of x.
    divisors = [[] for _ in range(maxA+1)]
    for d in range(1, maxA+1):
        for j in range(d, maxA+1, d):
            divisors[j].append(d)

    # The main idea:
    # We want to sum, for each nonempty subsequence B of (A1,...,Am), its "score" which is
    #   score(B) = sum_{i=1}^{k-1} gcd(B_i, B_{i+1}).
    #
    # Every subsequence ending at some index can be built either as a single element (score = 0)
    # or by extending a subsequence from an earlier prefix by appending the new element.
    #
    # Suppose we have already processed elements 1..m-1.
    # Let F be the set of all nonempty subsequences from the prefix, and let for every such subsequence
    # S we denote its accumulated score as s(S) and its last element value as v.
    # When we extend S by A_m (call it x), the new subsequence S' has:
    #   score = s(S) + gcd(v, x)
    # and its last element becomes x.
    #
    # Now, if we let:
    #   gScore = (sum_{S in F} s(S))    and    gCount = (number of subsequences in F)
    # then the sum of scores of all new extended subsequences equals:
    #   sum_{S in F}[ s(S) + gcd(v, x) ] = gScore + (sum_{S in F} gcd(v, x) ).
    #
    # To efficiently compute sum_{S in F} gcd(v, x), observe that if we group by divisors,
    # note the identity:
    #   gcd(v, x) = sum_{d|gcd(v,x)} phi(d).
    # Thus, if we maintain an array p[d] = (total count of subsequences S in F whose last element v is divisible by d),
    # we can write:
    #   sum_{S in F} gcd(v, x) = sum_{d|x} phi(d) * (sum_{v: d divides v} (# of S with last element = v))
    #                              = sum_{d in divisors[x]} phi(d) * p[d].
    #
    # Let newCount be the total number of new subsequences ending at x.
    # Since we can extend any subsequence from F and also start a new subsequence [x],
    # we have: newCount = gCount + 1.
    # The sum of scores for these new subsequences is:
    #   newScore = gScore + (sum_{d in divisors[x]} phi(d) * p[d]).
    #
    # Then, update the overall global accumulators:
    #   gCount += newCount   and   gScore += newScore.
    # Also, since these new subsequences end with x, for every divisor d of x,
    # add newCount to p[d].
    #
    # Finally, note that every nonempty subsequence in the prefix 1..m ends at its greatest index,
    # so the total sum of scores for the prefix equals gScore.
    
    # p[d] will store the sum over counts for those subsequences whose last element is divisible by d.
    p = [0] * (maxA+1)
    gCount = 0   # total count of subsequences from previous prefix
    gScore = 0   # total accumulated score for previous prefix

    out_lines = [None] * n
    for i in range(n):
        x = A[i]
        # new subsequences ending with x are built by extending any subsequence from before or starting fresh.
        newCount = (gCount + 1) % mod

        extra = 0  # will accumulate sum_{S in F} gcd(v, x)
        for d in divisors[x]:
            extra = (extra + phi[d] * p[d]) % mod
        newScore = (gScore + extra) % mod

        # Update global counts with these new subsequences.
        gCount = (gCount + newCount) % mod
        gScore = (gScore + newScore) % mod

        # Update p[d] for every divisor d of x: add the new subsequences that end with x.
        for d in divisors[x]:
            p[d] = (p[d] + newCount) % mod

        # For prefix m = i+1, answer is the sum of scores of all subsequences from the prefix.
        out_lines[i] = str(gScore)
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()