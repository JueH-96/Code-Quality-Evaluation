# YOUR CODE HERE
def can_rearrange(a, b):
    return sorted(a) == sorted(b)

N, Q = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

for _ in range(Q):
    l, r, L, R = map(int, input().split())
    sub_A = A[l-1:r]
    sub_B = B[L-1:R]
    
    if len(sub_A) != len(sub_B):
        print("No")
    else:
        print("Yes" if can_rearrange(sub_A, sub_B) else "No")