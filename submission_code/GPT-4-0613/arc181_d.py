import sys
from operator import itemgetter

def main():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    M = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    P = [(p, i) for i, p in enumerate(P)]
    P.sort(key=itemgetter(0))

    inv = [0]*(N+1)
    bit = [0]*(N+1)
    for i in range(N):
        x = P[i][1] + 1
        while x <= N:
            bit[x] += 1
            x += x & -x
        x = P[i][1] + 1
        s = 0
        while x:
            s += bit[x]
            x -= x & -x
        inv[P[i][0]] = i+1 - s

    acc_inv = [0]*(N+1)
    for i in range(1, N+1):
        acc_inv[i] = acc_inv[i-1] + inv[i]

    ans = [0]*M
    j = 0
    for i in range(1, N+1):
        while j < M and A[j] == i:
            ans[j] = acc_inv[N] - 2*acc_inv[i] + i*(i-1)
            j += 1

    print('
'.join(map(str, ans)))

if __name__ == "__main__":
    main()