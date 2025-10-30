N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Check if A and B are good sequences
for i in range(M):
    if A[i] == B[i]:
        print("No")
        exit()

# Check if A and B have the same sequence of 0s and 1s
if sorted(A) == sorted(B):
    print("Yes")
else:
    print("No")