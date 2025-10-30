def solve():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().strip().split()
    N, K = map(int, data)

    # We need to find all "good" sequences of length N*K in which each integer 1..N
    # appears exactly K times, ordered lexicographically. Let S be the number of such
    # sequences. We want the floor((S+1)/2)-th such sequence in lexicographic order.
    #
    # A standard combinatorial fact: The number of permutations of a multiset with
    # counts freq[1], freq[2], ..., freq[N] (where sum(freq) = M) is:
    #      M! / (freq[1]! * freq[2]! * ... * freq[N]!)
    #
    # Here freq[i] = K for i=1..N, so initially the total number of good sequences is
    #    S = (N*K)! / (K!)^N.
    #
    # To find the "k-th" permutation (here k = floor((S+1)/2)) of this multiset in
    # lex order, one can do the following well-known step-by-step approach:
    #
    #  1. Let current multiset frequencies be freq[] and let "remaining" = sum(freq).
    #  2. Compute the total number W of permutations of freq[] (call this ways(freq)).
    #  3. For each possible next element x (from smallest to largest) for which freq[x]>0:
    #       - Let W_x = ways(freq with freq[x] decreased by 1)
    #       - If W_x >= k, then the next element of our desired sequence is x;
    #            set freq[x] -= 1, ways(freq) = W_x, and move to pick the next position.
    #         else k -= W_x (we skip these W_x permutations) and continue to the next x.
    #
    # However, directly computing factorials for large N*K can be huge.  In Python,
    # we can still implement a "textbook" version that uses big integers and hope
    # it works for smaller or moderate inputs.  The idea is:
    #
    #   ways(freq) at any point can be updated by a ratio:
    #       ways(freq with freq[x]-1) = ways(freq) * freq[x] / remaining
    #
    # so we do not re-compute factorials from scratch each time.  But we *do* need
    # an initial ways(freq) = S.  A straightforward (but memory-heavy) way is to
    # precompute factorials up to (N*K).  This can be very large (up to 250000),
    # but in a theoretical setting (or with large memory/time), it is the most
    # direct solution.  For truly large N*K, one would need a more advanced approach.
    #
    # Below is the direct implementation of the median-finding procedure using
    # precomputed factorials and big-integer arithmetic in Python.

    from math import factorial

    # freq[i] will store how many times integer (i+1) still remains to be placed.
    freq = [K]*N
    total_len = N*K  # length of the final sequence

    # Precompute factorials from 0..total_len.
    # Note: This can be very large for N,K up to 500, but is the straightforward way.
    fact = [1] * (total_len + 1)
    for i in range(1, total_len + 1):
        fact[i] = fact[i - 1] * i

    # Compute S = (N*K)! / (K!)^N  (the total number of good sequences).
    denom = 1
    for _k in freq:
        denom *= fact[_k]
    S = fact[total_len] // denom  # big integer

    # We want M = floor((S+1)//2).
    M = (S + 1) // 2

    # We'll build the answer step by step.
    answer = []
    current_ways = S  # ways(freq) at the current step
    remaining = total_len

    for _pos in range(total_len):
        # We try each candidate i (1..N), in ascending order, to see if we pick i here.
        for i in range(N):
            if freq[i] == 0:
                continue
            # The number of permutations if we fix the next element as (i+1):
            #   partial = ways(freq) * freq[i] / remaining
            # (because ways(freq with freq[i]-1) = ways(freq) * freq[i] / remaining)
            partial = (current_ways * freq[i]) // remaining

            if partial >= M:
                # We choose i+1 as the next element
                answer.append(i + 1)
                freq[i] -= 1
                # Update current_ways and break:
                current_ways = partial
                remaining -= 1
                break
            else:
                # Skip those partial permutations and decrement M
                M -= partial
        else:
            # This theoretically should not happen if inputs are consistent
            # and M is in the correct range.
            pass

    print(" ".join(map(str, answer)))


def _main():
    solve()

# If you want to run solve() directly (instead of _main()),
# you can uncomment the next line:
# solve()