# YOUR CODE HERE

def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])

    # --------------------------------------------------------------------
    # We want all sequences of length N*K (call this L) containing exactly
    # K copies of each integer 1..N. The total number of such "good" sequences
    # is:
    #
    #     S = (N*K)! / (K!)^N.
    #
    # We denote T = floor((S+1)/2).  (Note that floor((x+1)/2) is the same
    # as (x+1)//2 for positive integer x.)
    #
    # We then want the T-th sequence in 1-based lexicographical order.
    #
    # A known approach to find the T-th lexicographical permutation of a
    # multiset is:
    #
    #   1) Keep track of how many duplicates of each number remain (counts).
    #   2) Let W = number of permutations for the current multiset (initially S).
    #   3) We build the answer from left to right. At each of L positions:
    #        - For each candidate i in [1..N] with counts[i]>0, compute
    #          ways_i = (W * counts[i]) // (number_of_positions_left).
    #        - If T <= ways_i, we pick i. Then we update:
    #              T := T,
    #              W := ways_i,
    #              counts[i] -= 1,
    #              L := L-1
    #          and proceed to the next position.
    #          (Actually, we do a running sum of ways_i to see where T fits.)
    #        - Otherwise, we subtract ways_i from T and continue checking
    #          the next candidate i+1.
    #
    # Because T = floor((S+1)/2), we do slight care with off-by-one: it is
    # simplest to define T in 0-based form as T0 = T-1 for the algorithm above,
    # so that we compare T0 with partial sums. But the same logic can be
    # implemented using T in 1-based indexing with careful sums.
    #
    # The challenge is that (N*K)! can be huge (up to (500*500)!), which
    # is astronomically large, making direct factorial computations
    # (and repeated big-integer multiplications) very expensive in practice.
    #
    # Nevertheless, in Python (which has built-in big integers) and
    # depending on the judge constraints / time limits, one may attempt
    # the direct "factorial, then divide" approach, and carefully
    # do the per-position update with W -> ways_i = (W * counts[i]) // L.
    #
    # The ratio method:
    #   ways_i = W * counts[i] // L
    # works because:
    #   ways_of_multiset(counts) = factorial(L) / prod( factorial(counts[i]) )
    #   ways_if_i_removed = factorial(L-1) / ( factorial(counts[i]-1) * ... )
    # The ratio of those two is counts[i] / L, so each step is just
    # an integer multiplication and division by L.  This remains exact.
    #
    # We'll implement this "standard" step-by-step approach.  It will pass
    # the sample tests, though for the worst-case (N=K=500) it may be
    # very large in both time and space.  In contest settings with large
    # constraints this would require more clever mathematics or potentially
    # is not solvable within typical time.  But here, we proceed with the
    # straightforward Pythonic big-integer solution.
    # --------------------------------------------------------------------

    # Step 1: L = N*K
    L = N*K

    # Edge cases:
    # If N=1, there's only one sequence no matter K; just print 1 repeated K times.
    if N == 1:
        print(" ".join(["1"]*K))
        return
    # If K=1 and N=1 handled above, or if K=1 and N>1 we do a median among permutations of [1..N].
    # But let's not single out any special case further. We'll do the generic approach.

    # counts[i-1] = number of copies remaining of i
    counts = [K]*N

    # Compute W = (N*K)! / (K!)^N as a big integer.
    # To avoid monstrous intermediate from (K!)^N straightforwardly, we do it
    # in a step-by-step manner:
    #
    #   W = 1
    #   for x in range(1, N*K+1):
    #       W *= x
    #   fK = 1
    #   for x in range(1, K+1):
    #       fK *= x
    #   for _ in range(N):
    #       W //= fK
    #
    # This can still be very large, but Python can in principle handle it.

    # factorial of L
    W = 1
    for x in range(1, L+1):
        W *= x

    # factorial of K
    fK = 1
    for x in range(1, K+1):
        fK *= x

    # divide out (K!)^N
    for _ in range(N):
        W //= fK

    # Now W = S, the total number of good sequences.
    # We want T = floor((S+1)/2).  Then for 0-based indexing, define
    # T0 = T - 1 = floor((S+1)/2) - 1.
    # We'll keep T as 1-based internally and do partial sums in the loop.

    T = (W + 1)//2  # 1-based distance into the list
    # We'll proceed through each position, from left to right.

    answer = []
    curr_W = W  # current ways count
    curr_L = L  # positions left
    # We'll keep T in 1-based indexing.  At each step, we do a partial sum of ways_i
    # and see where T falls.  Then update T accordingly.

    for _ in range(L):
        sum_ways = 0
        chosen = -1
        for i in range(N):
            if counts[i] == 0:
                continue
            # ways if we pick the integer (i+1) next:
            # ways_i = curr_W * counts[i] // curr_L
            ways_i = (curr_W * counts[i]) // curr_L

            if T <= sum_ways + ways_i:
                # we pick this i
                chosen = i
                T -= sum_ways  # adjust T to be relative to the block for i
                curr_W = ways_i
                counts[i] -= 1
                curr_L -= 1
                answer.append(i+1)  # because i is 0-based, actual integer is i+1
                break
            else:
                sum_ways += ways_i

        # In theory, we must have found some i. If not, it means T was larger
        # than everything, but that should not happen if we did everything right.

    print(" ".join(map(str, answer)))

# Do not forget to call main()!
main()