import sys
from collections import Counter

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    count = Counter(A)
    squares = set([i**2 for i in range(2*10**5+1)])

    pairs = 0
    for i in range(N):
        for j in range(i+1, N):
            if A[i]*A[j] in squares:
                pairs += 1

    print(pairs)

solve()