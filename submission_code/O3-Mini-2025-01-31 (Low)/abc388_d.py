def main():
    import sys,sys
    data = sys.stdin.read().split()
    if not data: 
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # Explanation:
    # Every alien starts with A[i] stones (i from 1-indexed perspective).
    # They “mature” (become an adult) in year i.
    # When an alien becomes adult, every existing adult (i.e. aliens 1..i-1) that “has at least one stone at that moment” gives 1 stone.
    #
    # Because the giving is “greedy” (everyone with ≥1 stone gives a stone) and the events
    # occur sequentially (year 1, then year 2, …), we can show that each alien’s ability to give coins
    # is “consecutive”; that is, once an alien becomes an adult (year i) it will give stones in
    # successive events until it runs out. More precisely, if an alien i (1-indexed) ends up with X_i stones
    # at the moment it becomes adult (its own big day) then it will donate coins at every later event,
    # for a total of exactly d_i = min(X_i, n - i) donations.
    #
    # But note: X_i is not just A[i] – because as a minor it might later receive gifts before it’s
    # forced to donate. In fact, for alien i, let r_i be the number of gifts (stones) received
    # at the moment it becomes adult from donors with index less than i. Then:
    #
    #    X_i = A[i] + r_i
    #    d_i = min(X_i, n - i)
    #    B_i = final_stones for alien i = X_i - d_i.
    #
    # Next, what is r_i?
    # When alien i becomes adult, every alien j with 1 <= j < i gives a stone if that donor j has not yet given
    # as many gifts as possible when it encountered an event. Donor j gives at events j+1, j+2, …, j + d_j.
    # Thus, donor j gives at the event when alien i becomes adult if and only if:
    #         i <= j + d_j.
    #
    # So, for alien i:
    #      r_i = count of j in 1..(i-1) with j + d_j >= i.
    #
    # We want to compute r_i for i = 1 ... n in an online fashion.
    # We can define E_j = j + d_j.
    # Then, for each i, r_i = (i-1) minus the number of previous donors j with E_j < i.
    #
    # To efficiently count how many j in [1, i-1] have E_j < i we use a Fenwick tree (binary indexed tree)
    # over possible values of E_j. Note that:
    #    - For an alien j, d_j = min( A[j] + r_j, n - j ), so E_j = j + d_j is in the range [j, n].
    # We use 1-indexing for aliens and for the Fenwick structure (index range 1..n).
    
    size = n  # possible positions for E value: 1 up to n.
    fenw = [0] * (size + 1)
    
    def fenw_update(i, delta):
        # Update Fenw tree at index i by delta
        while i <= size:
            fenw[i] += delta
            i += i & -i

    def fenw_query(i):
        # Query cumulative frequency in [1,i]
        s = 0
        while i:
            s += fenw[i]
            i -= i & -i
        return s

    # We'll store for alien i (1-indexed):
    #    r[i]: gifts received when becoming adult
    #    d[i]: number of donations it gives afterward.
    #    E[i] = i + d[i]
    # And final answer for alien i:
    #    B_i = (A[i] + r[i]) - d[i]
    r_arr = [0] * (n + 1)
    d_arr = [0] * (n + 1)
    E_arr = [0] * (n + 1)
    B = [0] * (n + 1)
    
    # Process aliens in order (1-indexed).
    for i in range(1, n + 1):
        # r[i] = (i-1) minus count of j in 1..(i-1) for which E[j] < i.
        less = fenw_query(i - 1)
        r_arr[i] = (i - 1) - less
        x = A[i - 1] + r_arr[i]  # total stones at the moment alien i becomes adult
        avail = n - i  # number of donation opportunities (events after its own)
        # d[i] = how many donations alien i will give.
        d_arr[i] = x if x < avail else avail
        E_arr[i] = i + d_arr[i]
        # Update Fenw tree: add frequency at position E_arr[i].
        # E_arr[i] is in 1..n.
        fenw_update(E_arr[i], 1)
        # Final stones:
        B[i] = (A[i - 1] + r_arr[i]) - d_arr[i]

    # Print B_1, B_2, …, B_n (skip index 0)
    sys.stdout.write(" ".join(str(B[i]) for i in range(1, n + 1)))

if __name__ == '__main__':
    main()