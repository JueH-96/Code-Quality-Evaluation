# YOUR CODE HERE
def solve():
    import sys
    import sys
    def input():
        return sys.stdin.read()
    mod = 998244353
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    B = list(map(int, data[2:2+N]))
    q = B.count(-1)
    # Precompute pow(M, q, mod)
    pow_M_q = pow(M, q, mod)
    pow_M_q_1 = pow(M, q-1, mod) if q >=1 else 0
    pow_M_q_2 = pow(M, q-2, mod) if q >=2 else 0
    sum1 = 0  # B[i]==-1 and B[j]==-1
    sum2 = 0  # B[i]==-1 and B[j]!=-1
    sum3 = 0  # B[i]!=-1 and B[j]==-1
    sum4 = 0  # B[i]!=-1 and B[j]!=-1 and B[i] <=B[j]
    for i in range(N):
        Bi = B[i]
        for j in range(i+1, N):
            Bj = B[j]
            if Bi == -1 and Bj == -1:
                sum1 +=1
            elif Bi == -1 and Bj != -1:
                sum2 += Bj
            elif Bi != -1 and Bj == -1:
                sum3 += (M - Bi +1)
            elif Bi != -1 and Bj != -1:
                if Bi <= Bj:
                    sum4 +=1
    # Compute term1
    if q >=2:
        tmp = (M * (M +1) //2 ) % mod
        term1 = (sum1 * tmp) % mod
        term1 = (term1 * pow_M_q_2) % mod
    else:
        term1 =0
    # Compute term2
    if q >=1:
        term2 = (sum2 % mod) * pow_M_q_1 % mod
        term3 = (sum3 % mod) * pow_M_q_1 % mod
    else:
        term2 =0
        term3 =0
    # Compute term4
    term4 = (sum4 % mod) * pow_M_q % mod
    # Sum S
    S = (term1 + term2 + term3 + term4) % mod
    # Compute sum_f
    sum_f = (N * pow_M_q) % mod
    sum_f = (sum_f - S) % mod
    sum_f = (sum_f + mod) % mod
    print(sum_f)