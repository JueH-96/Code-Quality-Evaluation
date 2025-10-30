import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def find_good_sequences(N, A):
    total_score = 0
    for S1 in range(1, 1001):
        for S2 in range(1, 1001):
            if gcd(S1, S2) == 1 and f(S1 / S2) == A[0]:
                S = [S1, S2]
                for i in range(1, N-1):
                    S_i = S[-1]
                    S_ip1 = int(S_i * A[i] / f(S_i / S[i]))
                    if gcd(S_i, S_ip1) != 1:
                        break
                    S.append(S_ip1)
                else:
                    if len(S) == N and gcd(S[0], S[1], *S[2:]) == 1:
                        score = 1
                        for s in S:
                            score *= s
                        total_score = (total_score + score) % 998244353
    return total_score

def f(x):
    p, q = 1, 1
    while x != p / q:
        if x > p / q:
            p += 1
        else:
            q += 1
    return p * q

if __name__ == '__main__':
    N = int(input())
    A = [int(x) for x in input().split()]
    print(find_good_sequences(N, A))