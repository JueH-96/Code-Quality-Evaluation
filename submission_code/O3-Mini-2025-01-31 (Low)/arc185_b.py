def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    t = int(data[0])
    pos = 1
    results = []
    # Explanation:
    # We are given a sequence A and allowed to “transfer” one unit from a later index j
    # to an earlier index i (i < j). Hence, in each move A[i] increases and A[j] decreases.
    # Thus, on one hand we are free to increase early values arbitrarily and decrease later
    # values arbitrarily; on the other hand the total sum is fixed.
    #
    # It turns out that the necessary and sufficient condition is that one can “redistribute”
    # the sum into a non-decreasing sequence B (with B1 <= B2 <= ... <= BN) subject to the
    # following observation:
    #   – There is no restriction on how much units we add/deduct from a particular index,
    #     except that the overall sum remains the same.
    #   – In any achievable final sequence B we can think of having “lifted up” some entries
    #     (by adding units) and “lowered” some entries (by subtracting units) so that
    #     B_i can be made arbitrarily high provided we do transfers from right indices (and
    #     similarly can be arbitrarily low if we give away units) but with the subtle constraint:
    #     the net effect of the allowed moves is that the only “allowed” re‐distribution is one
    #     that transfers mass from right (later) positions to left (earlier) positions.
    #
    # Hence one may show (via a “greedy” reconstruction argument) that a necessary and sufficient
    # condition is that the following feasibility test holds:
    #
    # Suppose the total sum S = sum(A). For any candidate final non‐decreasing sequence B with
    # sum S, one may ask: what is the “minimum” B that is non‐decreasing? Since we are free to choose
    # the B-values (apart from the total sum S) if we try to “minimize” B point–wise, we will try to
    # equalize as much as possible.
    #
    # In fact one may check that if we let 
    #      base = S // N    and    r = S mod N,
    # then the lexicographically smallest (and hence minimum in the prefix–sum sense)
    # non-decreasing sequence with sum S is given by:
    #
    #      B_1 = B_2 = ... = B_{N-r} = base    and 
    #      B_{N-r+1} = ... = B_N = base + 1.
    #
    # Let P_min[k] be its prefix sum for k=1,...,N. Then a necessary and sufficient condition for being able 
    # to “reach” a non-decreasing sequence using the allowed operations is that for every k,
    # the prefix sum of the original array is no more than the prefix sum of B:
    #
    #      for every k from 1 to N,   sum(A[1..k]) <= P_min[k].
    #
    # (One may prove that if this holds then we can “transfer” mass from later indices and achieve any 
    # target non-decreasing sequence B (in particular, one may “raise” an early entry arbitrarily)
    # while keeping the invariant that the cumulative mass in the first k positions is at least what
    # it was originally.)
    #
    # For example, in Sample Input 1, test case 1:
    #    A = [1, 7, 5], S=13, N=3, so base = 4 and r = 1.
    # The minimal non-decreasing B is: [4, 4, 5] with prefix sums [4, 8, 13].
    # The prefix sums of A are [1, 8, 13]. Since 1<=4,8<=8,13<=13 the condition holds.
    #
    # For test case 2: A = [9, 0], S=9, N=2, so base = 4 and r = 1.
    # The minimal non-decreasing B is: [4, 5] (prefix sums [4,9]). But A’s prefix sum is [9,9]; since 9 > 4,
    # it is impossible.
    #
    # The following implementation checks this condition for every test case.
    #
    out_lines = []
    for _ in range(t):
        n = int(data[pos])
        pos += 1
        A = list(map(int, data[pos:pos+n]))
        pos += n
        S = sum(A)
        base = S // n
        r = S % n
        # Build the minimal non-decreasing sequence's prefix sums.
        # For first (n-r) positions: each is equal to base.
        # For last r positions: each is equal to base+1.
        prefix_min = [0]*(n+1)
        for i in range(1, n+1):
            if i <= n - r:
                prefix_min[i] = prefix_min[i-1] + base
            else:
                prefix_min[i] = prefix_min[i-1] + (base+1)
        possible = True
        running = 0
        for i in range(1, n+1):
            running += A[i-1]
            if running > prefix_min[i]:
                possible = False
                break
        out_lines.append("Yes" if possible else "No")
    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()