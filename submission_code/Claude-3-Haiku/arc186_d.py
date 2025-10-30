from collections import defaultdict
from math import comb

MOD = 998244353

def count_polish_sequences(A, N):
    dp = defaultdict(int)
    dp[(0,)] = 1

    for i in range(N):
        for j in range(A[i]+1):
            dp[(i+1,) + tuple(range(j))] = (dp[(i+1,) + tuple(range(j))] + dp[(i,) + tuple(range(A[i]-j+1))]) % MOD

    return sum(dp.values()) % MOD

if __name__ == "__main__":
    N = int(input())
    A = [int(x) for x in input().split()]
    print(count_polish_sequences(A, N))