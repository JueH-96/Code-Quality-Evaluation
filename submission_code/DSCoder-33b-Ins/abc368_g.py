# YOUR CODE HERE

import sys

def solve():
    n = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    q = int(input())

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            A[query[1]-1] = query[2]
        elif query[0] == 2:
            B[query[1]-1] = query[2]
        else:
            l, r = query[1]-1, query[2]-1
            v = 0
            for i in range(l, r+1):
                if i == l or A[i] + v > v * B[i]:
                    v += A[i]
                else:
                    v *= B[i]
            print(v)

solve()