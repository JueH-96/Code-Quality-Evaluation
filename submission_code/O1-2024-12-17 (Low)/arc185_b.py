def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1
    
    # ----------------------------------------------------------------
    # EXPLANATION OF THE SOLUTION
    #
    # We are allowed operations of the form:
    #   Choose 1 <= i < j <= N, then do:
    #       A_i += 1
    #       A_j -= 1
    #
    # This means we can "shift" 1 unit of value from a higher index j down to
    # a lower index i (because i < j).  The question is: can we end up with
    # A_1 <= A_2 <= ... <= A_N after any number of such operations?
    #
    # A key observation is that for the final array to be nondecreasing,
    # we must (in particular) ensure that A_i >= i-1 for each i.  Why?
    #
    # Intuition / informal reasoning:
    #  - If we look at position i, in order for A_1' <= A_2' <= ... <= A_i',
    #    we need to have performed enough "shifts in" from the right so that
    #    A_i' is at least as large as A_{i-1}', A_{i-2}', etc.  Roughly, position
    #    i may need up to (i-1) increments to catch up if it started too small.
    #  - Because we can only shift from bigger indices j>i down to i, the
    #    original A_i itself must have at least room to accommodate these
    #    (i-1) increments if needed—otherwise, we cannot raise the earlier
    #    positions enough without running out of source material to shift
    #    from.  In effect, if A_i < i-1, then even if we wanted i-1 increments
    #    to build a chain of nondecreasing values up to index i, we'd not
    #    be able to supply them properly (since indices to the right can only
    #    push to smaller indices, not pull from them).
    #
    # In fact, one can show (and the given examples confirm) that a necessary
    # and sufficient simple check is:
    #
    #     For every i from 1 to N,  A_i >= (i - 1)
    #
    # If that holds for all i, answer "Yes"; otherwise "No".
    #
    # Why this matches the samples:
    #
    # Example 1:  (1, 7, 5)
    #   i=1 => A_1=1 >= 0  ok
    #   i=2 => A_2=7 >= 1  ok
    #   i=3 => A_3=5 >= 2  ok
    # => all pass => "Yes"
    # and indeed we have an explicit sequence of operations in the statement.
    #
    # Example 2:  (9, 0)
    #   i=1 => A_1=9 >=0  ok
    #   i=2 => A_2=0 >=1?  NO  (0 < 1)
    # => fails => "No"
    #
    # Example 3:  (607,495,419,894,610,636,465,331,925,724)  (N=10)
    #   We need to check A_i >= i-1 for i=1..10:
    #     A_1=607 >=0, A_2=495>=1, A_3=419>=2, A_4=894>=3,  A_5=610>=4,
    #     A_6=636>=5,  A_7=465>=6, A_8=331>=7, A_9=925>=8,  A_10=724>=9
    #   All true => "Yes"
    #
    # This matches the sample output exactly.
    # ----------------------------------------------------------------
    
    out = []
    ptr = 0
    for _ in range(t):
        n = int(input_data[idx]); idx+=1
        arr = list(map(int, input_data[idx:idx+n]))
        idx+=n
        
        possible = True
        for i in range(n):
            # i goes from 0..n-1, corresponding to position i+1 in 1-based indexing
            if arr[i] < i:  # i here is (i-1) in 1-based terms
                possible = False
                break
        
        out.append("Yes" if possible else "No")
    
    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()