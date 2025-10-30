# YOUR CODE HERE
from collections import Counter

def solve():
    N, Q = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    for _ in range(Q):
        l, r, L, R = map(int, input().split())
        l -= 1
        r -= 1
        L -= 1
        R -= 1

        A_subsequence = A[l:r+1]
        B_subsequence = B[L:R+1]

        if Counter(A_subsequence) == Counter(B_subsequence):
            print("Yes")
        else:
            print("No")

solve()