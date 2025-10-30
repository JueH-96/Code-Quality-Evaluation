# YOUR CODE HERE
import sys
import threading

MOD = 998244353

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:]))

    N_inv = pow(N, MOD -2, MOD)

    # We will compute expected salary E[salary] = sum_{i=1}^N A_i * E[P_i]
    # where E[P_i] is the probability that A_i is paid

    # Since the process can only reach x=i by rolling y=i when x<i,
    # and x increases from 0, and the process terminates when y<=x.
    # The probability to reach x=i is P_i
    # P_i = (N-1)/(N)^(i-1) * (1/N)

    # So we can compute P_i recursively:
    P = [0] * (N +1)
    P[0]=1
    E_salary = 0

    for i in range(1,N+1):
        # Probability that A_i is paid
        # P_i = P[i -1] * (N - (i -1)) * N_inv * N_inv
        if i==1:
            P_i = N_inv
        else:
            P_i = P_i * (N - (i -2) -1) * N_inv % MOD

        P[i] = P_i
        E_salary = (E_salary + A[i-1]*P_i)%MOD

    # Since we have fractions, we need to compute the rational expected value y/x modulo MOD
    print(E_salary % MOD)
    

threading.Thread(target=main).start()