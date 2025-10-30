def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1]
    
    # Convert S to a 1-based array of integers (0 or 1).
    # A[i] = 0 or 1, for i=1..N
    A = [0]*(N+1)
    for i in range(1, N+1):
        A[i] = 1 if S[i-1] == '1' else 0

    #------------------------------------------------------------
    # p[j] will store the index of the last '0' seen up to (and including) position j
    # If no zero up to j, p[j] = 0.
    # 1-based indexing.
    p = [0]*(N+1)
    for j in range(1, N+1):
        if A[j] == 0:
            p[j] = j
        else:
            p[j] = p[j-1]
    
    #------------------------------------------------------------
    # We will build prefix sums for "even-position bits" and "odd-position bits".
    # (Here "even" and "odd" refer to the 1-based index i%2.)
    # ps0[i] = sum of A[k] for k=1..i where k%2==0
    # ps1[i] = sum of A[k] for k=1..i where k%2==1
    ps0 = [0]*(N+1)
    ps1 = [0]*(N+1)
    for i in range(1, N+1):
        ps0[i] = ps0[i-1]
        ps1[i] = ps1[i-1]
        if i % 2 == 0:
            ps0[i] += A[i]
        else:
            ps1[i] += A[i]

    # A small helper to get the sum of A over [L..R] of those positions
    # whose parity is x (0 or 1):
    def sum_parity(x, L, R):
        # x=0 => use ps0, x=1 => use ps1
        if R < L:
            return 0
        return (ps0[R] - ps0[L-1]) if x == 0 else (ps1[R] - ps1[L-1])

    # Also we want to quickly count how many indices in [L..R] have parity x.
    # For 1-based indexing, the number of even indices up to i is i//2,
    # and the number of odd indices up to i is i - i//2.
    def count_even(L, R):
        # number of even indices in [1..X] = X//2
        # so in [L..R] = (R//2) - ((L-1)//2)
        if R < L:
            return 0
        return (R//2) - ((L-1)//2)
    
    def count_odd(L, R):
        # total = R-L+1
        # even = count_even(L,R)
        # odd = total - even
        if R < L:
            return 0
        total = R - L + 1
        return total - count_even(L,R)

    #------------------------------------------------------------
    # SUM OF FINAL STATES FOR ALL SUBSEQUENCES = sumNoZero + sumWithZero.
    #
    # 1) sumWithZero:
    #    If the subsequence i..j (T = S[i+1..j]) has at least one zero,
    #    then let p_j = last zero in [1..j]. We need p_j >= i+1 => i <= p_j-1.
    #    The final state is 1 if (j - p_j) is even, else 0.
    #    So for each j with p_j>0, there are (p_j - 1) valid i's.
    #    Contribution = (p_j - 1) if (j-p_j) even, else 0.
    #
    sum_with_zero = 0
    for j in range(1, N+1):
        if p[j] > 0:
            if ((j - p[j]) % 2) == 0:
                sum_with_zero += (p[j] - 1)
    
    # 2) sumNoZero:
    #    If T = S[i+1..j] has no zero, that means p[j] < i+1 => i >= p[j].
    #    If p[j]==0, then i can be 1..j.
    #    If p[j]>0, then i can be p[j]..j.
    #    For such subsequences, T is all 1's, so final state = A[i] XOR ((j-i) mod 2).
    #    Equivalently,
    #      final_state = A[i] if (j-i) is even
    #                   = 1 - A[i] if (j-i) is odd
    #
    #    But (j-i) even <=> i%2 == j%2  for 1-based indexing.
    #    So final_state = A[i] if i%2 == j%2
    #                    = 1 - A[i] if i%2 != j%2
    #
    #    We want sum_i( final_state ), for i in [start..j],
    #    where start = 1 if p[j]==0 else p[j].
    #
    sum_no_zero = 0
    for j in range(1, N+1):
        start = 1 if p[j] == 0 else p[j]
        if start > j:
            continue  # no valid i in this case
        e = j % 2   # the parity of j
        o = 1 - e   # the other parity
        # sum over i of A[i] for i%2 = e
        sum_same_parity = sum_parity(e, start, j)
        # sum over i of (1 - A[i]) for i%2 = o
        # = (# of i in that range with i%2=o) - sum of A[i] for i%2=o
        c_diff = count_even(start, j) if o == 0 else count_odd(start, j)
        s_diff = sum_parity(o, start, j)
        # if o == 0 => we are counting evens in [start..j]
        # if o == 1 => we are counting odds in [start..j]
        
        sub_no_zero = sum_same_parity + (c_diff - s_diff)
        sum_no_zero += sub_no_zero
    
    answer = sum_no_zero + sum_with_zero
    print(answer)

# Call main() at the end
if __name__ == "__main__":
    main()