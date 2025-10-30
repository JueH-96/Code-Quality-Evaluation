import sys

def soln():
    S = input()
    Q = int(input())
    Ks = [int(x) for x in input().split()]

    # Perform the operation 10^100 times
    for _ in range(100):
        T = ''.join(c.lower() if c.isupper() else c.upper() for c in S)
        S = S + T

    # Answer the queries
    ans = []
    for K in Ks:
        ans.append(S[K-1])
    print(' '.join(ans))

if __name__ == '__main__':
    soln()