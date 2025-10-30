# Read the input
N = int(input())
A = [int(x) for x in input().split()]

# Check if the sequence is a geometric progression
if N < 2:
    print("Yes")
else:
    ratio = A[1] / A[0]
    for i in range(1, N):
        if A[i] / A[i-1] != ratio:
            print("No")
            exit()
    print("Yes")