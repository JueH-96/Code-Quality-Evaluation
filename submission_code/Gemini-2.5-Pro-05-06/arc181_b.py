import sys

def solve():
    S = sys.stdin.readline().strip()
    X_str = sys.stdin.readline().strip()
    Y_str = sys.stdin.readline().strip()

    if not S: # S is an empty string
        sys.stdout.write("Yes
")
        return

    n0X = X_str.count('0')
    n1X = len(X_str) - n0X
    n0Y = Y_str.count('0')
    n1Y = len(Y_str) - n0Y
    
    A_coeff = n0X - n0Y
    B_coeff = n1Y - n1X
    
    if A_coeff == 0:
        sys.stdout.write("Yes
")
        return
    
    if B_coeff == 0: # A_coeff != 0 here
        sys.stdout.write("No
")
        return

    # A_coeff != 0 and B_coeff != 0 here
    if (A_coeff > 0 and B_coeff < 0) or \
       (A_coeff < 0 and B_coeff > 0):
        sys.stdout.write("No
")
        return
        
    # A_coeff, B_coeff both non-zero and have same sign.
    N_S = len(S)
    
    pi = [0] * N_S
    for i in range(1, N_S):
        j = pi[i-1]
        while j > 0 and S[i] != S[j]:
            j = pi[j-1]
        if S[i] == S[j]:
            j += 1
        pi[i] = j
    
    # len_p is length of primitive root of S
    # N_S > 0 since S is not empty. pi[N_S-1] is valid.
    len_p = N_S - pi[N_S-1] 
    if N_S % len_p != 0:
        len_p = N_S
    
    a_s_val = N_S // len_p

    if (a_s_val * A_coeff) % B_coeff == 0:
        sys.stdout.write("Yes
")
    else:
        sys.stdout.write("No
")

num_test_cases = int(sys.stdin.readline())
for _ in range(num_test_cases):
    solve()