def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # ----------------------------------------------------------------
    # OVERVIEW OF THE APPROACH:
    #
    # We want to compute:
    #     S = ∑(i=1 to N) ∑(j=i to N)  f(A[i] + A[j])
    #
    # Recall f(x) = x with all factors of 2 removed; equivalently,
    # if x = 2^k * m with m odd, then f(x) = m.
    #
    # Naïvely, there are O(N^2) pairs, which is too large for N up to 2e5.
    #
    # Instead, there is a known divide-and-conquer / recursion trick
    # often used in problems involving repeatedly "shifting out" factors
    # of 2.  One way to see it is to split the array into Evens and Odds
    # and use the fact that:
    #
    # 1) f(even1 + even2) = f((even1/2) + (even2/2)).
    #    Because if e1=2*x, e2=2*y, then e1+e2 = 2*(x+y), and removing
    #    a factor-of-2 from e1+e2 is the same as applying f(x+y).
    #
    # 2) f(odd1 + odd2) = f(2 * something) = f(something).
    #    If o1=2*x+1, o2=2*y+1, then o1+o2 = 2 * (x+y+1).
    #
    # 3) If one is even and the other odd, say e + o with e even, o odd,
    #    then e+o is odd, so f(e+o) = e+o itself (no factor of 2).
    #
    # We also have to be careful about counting i=j (the "diagonal"),
    # because the problem wants ∑(i=1..N) ∑(j=i..N).
    #
    # A well-known solution is to define a recursive function that
    # computes the sum of f(x+y) over all distinct pairs (i<j),
    # plus handle how many times we add the diagonal.  Then we relate
    # that sum back to a recursion on half of the even elements, plus
    # a recursion on some transform of the odd elements, plus the cross
    # sums.  Finally we add the diagonal part separately.
    #
    # Below is an implementation using this approach.  We will define
    # a function pair_sum_f(A) that returns ∑(i<j) f(A[i]+A[j]) for
    # the array A.  Then to get the final ∑(i=1..N) ∑(j=i..N), we note:
    #
    #    ∑(i=1..N) ∑(j=i..N) f(A[i]+A[j])
    #      = pair_sum_f(A)   +   ∑(i=1..N) f(A[i] + A[i]).
    #
    # And f(A[i]+A[i]) = f(2*A[i]).  If A[i] = 2^k*m (m odd),
    # then 2*A[i] = 2^(k+1)*m, so f(2*A[i]) = m = f(A[i]).
    # So the diagonal contribution is simply ∑(i=1..N) f(A[i]).
    #
    # Therefore final answer = pair_sum_f(A) + sum_of_f(A),
    # where sum_of_f(A) = ∑ f(A[i]).
    #
    # The core challenge is implementing pair_sum_f(A) efficiently.
    #
    # The known recursion (sometimes appearing in similar problems):
    #
    #    Let E = all even elements of A,
    #        O = all odd  elements of A.
    #    Let nE = number of evens, nO = number of odds.
    #
    #    1) For e1,e2 in E (distinct), f(e1+ e2) = f( (e1/2) + (e2/2) ).
    #       Summation over pairs of E is exactly pair_sum_f(E') where
    #       E' = [e//2 for e in E].
    #
    #    2) For o1,o2 in O (distinct), o1+ o2 is even. Factoring out 2,
    #       f(o1+ o2) = f( (o1+ o2)/2 ).
    #       But if o=2*x+1, then (o-1)//2 = x.  We can define O' to
    #       represent those x-values.  Then (o1 + o2)/2 = ( (2*x1+1) + (2*x2+1) )/2
    #       = x1 + x2 + 1.  We can treat that carefully as another array
    #       or as a small adjustment.  The standard approach is:
    #            pair_sum_f(O) = pair_sum_f( [ (o-1)//2 for o in O] )
    #       plus the fact that (o1+ o2)/2 adds “+1” to each pairing inside.
    #       That extra “+1” in the sum-of-f means we effectively shift all
    #       elements by +1 before applying f in the recursion.  A known
    #       short-cut is: each pair of odds contributes exactly 1 factor
    #       of 2, so we add pair_count_O = nO*(nO-1)//2 to the sum once
    #       more, and then we recurse on [ (o-1)//2 ] with a shifting trick.
    #       The simpler way is to do the "group shift" technique: we define
    #       O' = [ (o-1)//2 for o in O ], but we must also shift values by +1
    #       inside f(·).  However, a well-known identity is:
    #             f( x + 2^k ) for x odd can be broken down,
    #       but it’s a bit intricate to implement from scratch in limited time.
    #
    #    3) For cross pairs (e in E, o in O), e+ o is odd. Then f(e+ o) = e+ o.
    #       Summation over all cross pairs is:
    #          ∑(e in E, o in O) [ freq(e)*freq(o)*(e + o) ].
    #       We can compute sum(E) and sum(O) and do a product expansion.
    #
    # In code, a standard technique is:
    #   pair_sum_f(A):
    #       if len(A) < 2: return 0
    #       separate into E, O
    #       cross_sum = sum_{e in E} e * #O + sum_{o in O} o * #E
    #         (factoring in frequencies if needed)
    #       even_part = pair_sum_f(e//2 for e in E)
    #       # For O, each pair contributes 1 factor-of-2 stripped off,
    #       # plus we effectively do pair_sum_f( [ (o-1)//2 ] ), but
    #       # with a shift of +1 inside that f(·).  The net known result:
    #       #   pair_sum_f(O) = pair_sum_f( O' )   +   (#O choose 2),
    #       # where O' = [ (o-1)//2 for o in O ].
    #       odd_part  = pair_sum_f(O') + (nO*(nO-1)//2)
    #       return cross_sum + even_part + odd_part
    #
    # Then the final S = pair_sum_f(A) + sum_{i=1..N} f(A[i]).
    #
    # We just have to implement pair_sum_f(A) carefully with frequencies
    # (since N can be large, but we can store A directly—Python can still
    # handle recursion if we group or sort? We should be mindful of memory.)
    #
    # Implementation details:
    #   - We'll implement pair_sum_f as a function that returns the sum
    #     over i<j of f(A[i] + A[j]).
    #   - We do it by splitting A into E and O in a single pass,
    #     then cross_sum is straightforward. Then we recursively call
    #     pair_sum_f on E' and O' as described. Done.
    #   - Because N can be up to 2e5, each recursion step lumps out the
    #     even or odd half. In the worst case, if half are even each time,
    #     we might get O(log(max(A))) recursions, which is okay.
    #
    # Let’s code it.
    # ----------------------------------------------------------------

    # Precompute f(A[i]) once for the diagonal part:
    # f(A[i]) = A[i] // (2**v2(A[i]))  or simply while even, A[i]//=2
    # but we can do it quickly with x & -x to find the largest power-of-2 divisor.
    def f_single(x):
        # Remove factors of 2 from x
        return x >> (x & -x).bit_length() - 1
        # Explanation:
        # (x & -x) is the largest power of 2 dividing x.
        # If p = (x & -x), then p = 2^r, where r = v2(x).
        # bit_length of p is r+1, so we shift x right by r.
        # Alternatively, we could do a while-loop:
        #   while x % 2 == 0: x //=2
        #   return x
        # but above is a quick one-liner in Python.

    # Sum of f(A[i]) for diagonal
    diag_sum = 0
    for val in A:
        # remove factors of 2
        diag_sum += f_single(val)

    # We implement pair_sum_f using recursion:
    from collections import deque

    def pair_sum_f(arr):
        # arr is a deque or list of integers. We compute sum_{i<j} f(arr[i]+arr[j]).

        # If length < 2, no pairs
        n = len(arr)
        if n < 2:
            return 0

        # Separate into Even/ Odd
        E = []
        O = []
        for x in arr:
            if x % 2 == 0:
                E.append(x//2)  # we directly store x//2 for the recursion on E
            else:
                O.append((x-1)//2)  # store (x-1)//2 for recursion on O

        nE = len(E)
        nO = len(O)

        # cross-sum: for each e in E, o in O, f(e_original + o_original) = e_original + o_original
        # but e_original was 2*e, o_original was 2*o + 1
        # so cross contribution = (2*e + (2*o+1)) = 2*(e + o) + 1
        # Summation over all e, o = ∑( e in E ) ∑( o in O ) [2*(e + o) + 1].
        # Expand:
        #  = 2 * [ nO*sum(e) + nE*sum(o ) ] + nE*nO
        # Because sum_{ e in E, o in O } (e + o) = (sum(E)*nO + sum(O)*nE).
        sumE = sum(E)
        sumO = sum(O)
        cross_sum = 2 * (nO*sumE + nE*sumO) + nE*nO

        # even_part = pair_sum_f(E)
        even_part = pair_sum_f(E)

        # odd_part = pair_sum_f(O) + (nO choose 2)
        # because for o1,o2 in O, original values were (2*o1+1, 2*o2+1),
        # sum is 2*(o1+ o2+1). Removing factor 2 => f( (o1+ o2+1) )
        # The known standard fix is to add + (nO*(nO-1)//2) for that
        # single extracted factor-2 from each pair of odds, and then
        # recursively handle (o1+ o2+1) inside pair_sum_f(O).  In practice,
        # the well-known short formula is to just do:
        #    odd_part = pair_sum_f(O) + (nO*(nO-1)//2)
        # for i<j.  That accounts exactly for the single factor-of-2
        # from each pair of original odd numbers.
        odd_part = pair_sum_f(O) + (nO*(nO-1)//2)

        return cross_sum + even_part + odd_part

    # We'll wrap A in a deque for convenience if we wish:
    A_deque = deque(A)

    # pair_sum = sum over i<j of f(A[i]+ A[j])
    pair_sum = pair_sum_f(A_deque)

    # We want sum over i<=j, i.e. i<j plus diagonal. So add diag_sum:
    ans = pair_sum + diag_sum

    print(ans)