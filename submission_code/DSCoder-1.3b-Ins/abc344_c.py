N = int(input())
A = list(map(int, input().split()))
M = int(input())
B = list(map(int, input().split()))
L = int(input())
C = list(map(int, input().split()))
Q = int(input())
X = list(map(int, input().split()))

for i in range(Q):
    if sum(A[:i+1]) == X[i] or sum(B[:i+1]) == X[i] or sum(C[:i+1]) == X[i]:
        print("Yes")
    else:
        print("No")