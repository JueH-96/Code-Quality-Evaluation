from collections import Counter

N, Q = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

for _ in range(Q):
    l, r, L, R = map(int, input().split())
    a_sub = A[l-1:r]
    b_sub = B[L-1:R]
    
    if Counter(a_sub) == Counter(b_sub):
        print("Yes")
    else:
        print("No")