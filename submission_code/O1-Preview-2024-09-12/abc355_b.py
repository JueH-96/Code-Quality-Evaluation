# YOUR CODE HERE
N, M = map(int, input().split())
A = set(map(int, input().split()))
B = set(map(int, input().split()))

C = sorted(A.union(B))

for i in range(len(C) - 1):
    if C[i] in A and C[i+1] in A:
        print("Yes")
        break
else:
    print("No")